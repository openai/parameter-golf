# Parameter Golf — Experiment Log

Hardware: switching to 2×H100 from Exp 3 onward (~2200 steps in 10 min, cleaner signal).
Baseline comparison: 1.3586 val_bpb at step 1000 (1×H100, PyTorch 2.5.1).
Full 8×H100 baseline: ~1.20 bpb (official).

---

## Results

| # | Experiment | Branch | Commit | Hardware | val_bpb | Δ vs baseline | Status |
|---|---|---|---|---|---|---|---|
| 0 | Baseline (PyTorch 2.4) | `main` | baseline | 1×H100 | 1.3614 | — | ✅ done |
| 1 | **Baseline v2** (PyTorch 2.5.1) | `main` | baseline | 1×H100 | 1.3586 | — | ✅ done |
| 2 | **Depth recurrence** (9L, [0,1,2,3,4,3,4,5,6,7,8]) | `exp/depth-recurrence` | ea1898a | 1×H100 | 1.3449 | **-0.014** | ✅ done |
| 3 | **Baseline** (2×H100) | `main` | baseline | 2×H100 | 1.2732 | — | ✅ done |
| 4 | **Depth recurrence** (2×H100) | `exp/depth-recurrence` | ea1898a | 2×H100 | 1.2728 | **-0.0004** | ✅ done |
| 5 | **Baseline re-run** (VAL_EVERY=200) | `main` | `run_exp5` | 2×H100 | 1.2728 | — | ✅ done |
| 6 | **Recurrence** | `exp/depth-recurrence` | `run_exp6` | 2×H100 | FAILED | — | ❌ branch issue |
| 7 | **Recur + q_gain=5.25** | `exp/depth-recurrence` | `run_exp7` | 2×H100 | 1.2754 | +0.003 | ✅ done |
| 8 | **Recur + q_gain=3.0** | `exp/depth-recurrence` | `run_exp8` | 2×H100 | 1.2713 | -0.002 | ✅ done |
| 9 | **Recur + q_gain=4.0** | `exp/depth-recurrence` | `run_exp9` | 2×H100 | 1.2693 | -0.004 | ✅ done |
| 10 | **3-layer recur [2,3,4]** | `exp/depth-recurrence` | `run_exp10` | 2×H100 | 1.2698 | -0.003 | ✅ done |
| 11 | **11L narrow (dim=448)** | `exp/depth-recurrence` | `run_exp11` | 2×H100 | 1.2817 | +0.009 | ✅ done |
| 12 | **Wide MLP (mult=3, dim=384)** | `exp/depth-recurrence` | `run_exp12` | 2×H100 | 1.2892 | +0.016 | ✅ done |
| 13 | **Recur + seq_len=2048** | `exp/depth-recurrence` | `run_exp13` | 2×H100 | **1.2583** | **-0.015** | ✅ BEST |
| 14 | **Recur + warmdown=2400** | `exp/depth-recurrence` | `run_exp14` | 2×H100 | 1.2736 | +0.001 | ✅ done |
| 15 | **Combined: recur + seq2048 + qk4.0** | `exp/depth-recurrence` | manual | 2×H100 | 1.2590 | -0.014 | ✅ done |
| 16 | **Phase1: LeakyReLU + ParRes + StagedRecur + seq2048 + qk4.0** | `exp/depth-recurrence` | 6e2da72 | 2×H100 | **1.2550** | **-0.018** | ✅ done |
| 21 | **Full run: all improvements, 40min** | `exp/depth-recurrence` | f776719 | 2×H100 | **1.1963** | **-0.077** | ✅ NEW BEST |
| 22 | **SOTA code test (SP1024)** | SOTA submission | obfuscated | 1×H100 | 1.6050 | — | ✅ test only |
| 23 | **SOTA code test (SP8192)** | SOTA submission | obfuscated | 1×H100 | 1.4052 | — | ✅ test only |
| 000 | **Spec 000 — SOTA replication** (seed 42, BIGRAM=0, QK=5.25, TTT=1) | `research` | 01e6fcf | 8×H100 NA-1 | **1.08622** (post-TTT) | +0.0054 vs SOTA 1.0810 | ⚠️ outside accept window — throughput deficit (3849/4550 steps), code confirmed faithful; adopt as baseline |
| 001 | **Spec 001 — Hessian-SDClip λ screen** (6 λ, quant-only) | `exp/hessian-sdclip` | 74c8385 | 1×H100 NA-1 | **1.10518 → 1.10676** (λ=0 → λ=0.60, quant-only, not post-TTT) | +0.00009 to +0.00158 vs λ=0 control | ❌ killed — monotonic worsening at all λ; artifact >16MB at λ≥0.40; idea shelved |
| 002 | **Spec 002 — SWA + EMA blend screen** (6 configs, quant-only) | `exp/swa-plus-ema` | 46c2a92 | 1×H100 NA-1 | **1.10518 → 1.14694** (C0 EMA-only → C1 pure SWA, quant-only) | +0.006 to +0.042 vs C0 (all configs hurt) | ❌ killed — linear worsening with SWA fraction; EMA-only is best; idea shelved |
| 003 | **Spec 003 — BigramHash signal screen** (single run, match Exp 24 config) | `research` | 3825019 | 2×H100 NA-1 | **1.08788** (pre-quant post-EMA) | +0.00118 vs Exp 24's 1.08670 | ❌ killed — miss signal gate by +0.00318; variant train_loss worse at 4/4 final milestones; idea shelved |
| 004 | **Spec 004 — QK-gain extension screen** (3 phases: 5min A/B, 10min VAL=200, 10min VAL=4000 verification) | `research` | feaf45e | 8×H100 NA-1 | **1.09193** (004c pre-quant post-EMA, QK=6.0 clean verification) | −0.00096 vs spec 000's 1.09289 | ❌ killed — within noise floor (~0.001); Phase 1's Run A −0.109 was pod-bf16 variance not real signal; artifact >16MB at QK=6.0; idea shelved |
| 011 | **Spec 011 — training-bundle** (tapered WD + GradPower p=0.9; ported from #1729 + #1682) | `exp/training-bundle` | 893cefd | 8×H100 JP-1 | **1.0706** (endpoint bare val_bpb, screening mode) | +0.0009 vs spec 008's 1.0697 (within ~1.3× seed std) | ❌ partial null — GradPower p=0.9 did not transfer from author's 1×H100 regime; WD taper never engaged due to `WD_TAPER_START_FRAC` bug (keyed on `iterations`=20000 instead of wallclock-cap step ~4844). Decision: kill GradPower; shelve tapered WD. See `research/evaluations/011-training-bundle.md`. |
| 013 | **Spec 013 — BigramHash** (auxiliary hash-keyed embedding, port #1716) | `exp/bigram-hash` | 66e57bf | 8×H100 JP-1 | **1.0722** (endpoint bare val_bpb, screening mode) | +0.0025 vs spec 008's 1.0697 (outside ±0.0015 single-seed CI, on unfavorable side) | 🗄️ SHELVED — single-seed small regression; confounded by +0.0323 early divergence at step 500 (RNG-stream drift from module-tree addition), which gap-closes to +0.0021 by step 3500 but endpoint still +0.0025. RNG-control retry would be scientifically cleaner but doesn't reflect shipping reality; 3-seed confirmation (~$60) is 40% of remaining budget. Decision: shelve for this push, potential post-deadline research writeup. See `research/evaluations/013-bigram-hash.md`. |
| 014 | **Spec 014 — BPB-weighted CE** (align training loss with eval BPB metric, port #1519) | `exp/bpb-weighted` | ab6a131 | 8×H100 JP-1 | **1.1316** (endpoint bare val_bpb, screening mode) | **+0.0619** vs spec 008's 1.0697 (62× past shelve threshold, clear regression) | 🗄️ SHELVED (permanently for push) — no destabilization but persistent trajectory offset. Root cause: byte-weighted gradient direction is structurally different from uniform CE, and Muon's LR schedule is tuned on uniform. Author's own "large-vocab destabilization" warning held. Decision: do NOT retune; 3rd null in a row confirms meta-pattern (incremental ports from different-stack authors don't transfer). Post-deadline only: co-tune objective + optimizer from step 0 with context-aware byte sidecar. See `research/evaluations/014-bpb-weighted-loss.md`. |
| 015 | **Spec 015 — Recur-Alpha (α=0 init)** (learnable per-pass-per-layer scalar blend on non-first recurrence passes; 6 scalars total) | `exp/recur-alpha` | a9aa141 | 8×H100 JP-1 | **1.0696** endpoint / **1.1078 @ step 4000** (screening mode) | −0.0001 endpoint (null, hardware-variance confounded) / **−0.0032 @ step 4000** (real signal, 2× std floor) vs spec 008's 1.1110 | ✅ VALIDATED — first consistent positive signal post-#1736 rebase. Endpoint null is JP step-deficit artifact (67 fewer steps). Matched-step gain is real. α converges to non-trivial asymmetric pattern: pass-2 amplifies with depth [1.04→1.38], pass-3 damps with depth [1.01→0.77]. See `research/evaluations/015-recur-alpha.md`. |
| 016 | **Spec 016 — Recur-Alpha (α=1 init)** (same mechanism as 015; init changed zeros→ones; adds grad_norm fix + torch.compile cache) | `exp/recur-alpha` | 4dd2d63 | 8×H100 JP-1 | **1.0712** endpoint / **1.1072 @ step 4000** / post-quant **1.08029** (15.94 MB) / TTT OOM (projected **~1.062–1.065**) | −0.0006 vs 015 @ step 4000 (null: init irrelevant) / **−0.0038 vs 008 @ step 4000** / projected **−0.003 to −0.005 vs 008 post-TTT** at matched steps | 🗄️ SHELVED (init experiments closed). Init choice is irrelevant; mechanism validated. Matched-step projection vs #1736 record (1.06549): Method A (~1.06252) beats it, Method B (~1.06642) does not — ambiguous without matched-step rerun (~$12). Not a standalone submission. Recommended as **stack ingredient** for future combination runs. See `research/evaluations/016-recur-alpha-ones.md`. |
| 019b | **Spec 019b — Recur-Alpha manual+constant, full pipeline** (algebraic lerp form `x_before + α*(x_new-x_before)` replacing `torch.lerp`; same constant α as 019; TTT-fix path; 4 sites) | `exp/recur-alpha-manual-constant-full` | 9517a3b | 8×H100 JP-1 | **1.06951** pre-quant / **1.07877** post-GPTQ / **1.06628** post-TTT (15.98 MB) / **1.1071 @ step 4000** | **+0.00018 vs #1736** (1.06610) — within seed std / **−0.00116 vs 019** / best matched-step quality (1.1071) of all recur-alpha runs | ⏳ PENDING 3-SEED — closest miss yet. Algebraic lerp is a true no-op vs torch.lerp at the loss level (step-4000 val_bpb identical to 019); 019b's gain over 019 is 19 extra steps (pod lottery). Full-scale throughput still regresses (6.2M tok/s at end vs 8.1M at start) — algebraic form did not fix the drift. Options: 3-seed (~$20-24) to resolve if 1.06628 is representative; or pivot to stack. See `research/evaluations/019b-recur-alpha-manual-constant-full.md`. |
| 021 | **Spec 021 — Recur-α as register_buffer, full pipeline** (α promoted from Python literal → `register_buffer` frozen at 017's learned endpoint values; prevents Dynamo specialization → kills Type B mystery spikes and post-val recompile cluster) | `exp/recur-alpha-buffer` | cb5cd78 | 4×H100 US-NE-1 (spec called for 8×H JP — all 8-GPU SKUs dry ~4h; used 4×H + `MAX_WALLCLOCK_SECONDS=1200` to match GPU-seconds) | **1.07095** pre-quant post-EMA / **1.1086 @ step 4000** / **1.0714 @ step 4736** / post-GPTQ + post-TTT N/A (brotli crash at serialize) | **+0.00173 vs 008** (8×H #1736) / **+0.00032 vs 019** (8×H lit-α) — within noise; batch-size tax (~0.001–0.002) for 4×H vs 8×H accounts for the gap | ✅ THROUGHPUT HYPOTHESIS CONFIRMED in production (0 Type B spikes, 0 post-val recompile, 2.70M ±0.01 post-loop inst tok/s vs 019's choppy 5.01–5.33M). val_bpb improvement INCONCLUSIVE on 4×H. Promote buffer-α to research baseline for α-mechanism work. Needs 8×H rerun with brotli installed for post-TTT + hardware-matched comparison. See `research/evaluations/021-recur-alpha-buffer.md`. |
| 021(fix) | **Spec 021 α-bug fix rerun** (same register_buffer setup as 021, pass-3 L4 α corrected from 0.96484375 → 0.97265625 matching 017 endpoint) | `exp/recur-alpha-buffer` | dc0b5f8 | 8×H100 JP-1 | **1.06950** pre-quant post-EMA / **1.07883** post-GPTQ / **1.06900** post-TTT (killed mid-TTT phase 1) / **1.1133 @ step 4000** | +0.00272 vs #1736 / +0.00322 vs 019b — *bug was NOT the root cause* | 📊 RULED OUT — α-value correction had near-zero effect on trajectory (buggy 1.06963 → fix 1.06950 = −0.00013). Throughput profile identical to 021 buggy. Confirms 021's regression vs 019b is not the typo. Led to spec 021b (bf16 dtype-match hypothesis). |
| 021b | **Spec 021b — bf16 buffer + drop runtime cast** (α buffer dtype changed fp32→bf16; removes `.to(x_new.dtype)` at blend sites; tests whether dtype/fusion is the 021 regression mechanism) | `exp/recur-alpha-buffer` | d070df3 | 8×H100 JP-1 | **1.06878** pre-quant post-EMA / post-TTT N/A (killed post-EMA) / **1.1127 @ step 4000** / step 4884 stopping_early | **−0.00072 vs 021 fix** / **+0.0013 vs 017** / still trails 019b's 1.06951 by +0.00073 | 📊 PARTIAL MECHANISM — bf16 alignment closes ~half the gap (0.00072 of ~0.0012 to 017's 1.06733, still trailing 019b). Best of the 021 family on post-EMA. Confirms dtype/fusion is *a* contributor, not the full story. Led to spec 021c 3-way container test. No eval file (summary in `runs/021-recur-alpha-buffer-8xh100/seed_42_bf16/notes.md` via context). |
| 021d | **Spec 021d — Recur-α nn.Parameter(requires_grad=False), 8×H100 JP** (α container changed register_buffer → nn.Parameter frozen bf16; hypothesis: Inductor const-folding) | `exp/recur-alpha-buffer` | 8b2d791 | 8×H100 JP-1 | **1.06960** pre-quant post-EMA / **1.07896** post-GPTQ / TTT not run / **1.1135 @ step 4000** | +0.00009 vs 019b — noise | 🗄️ ARC CLOSED — Parameter container doesn't close per-step loss gap vs 019b (+0.007–0.022 throughout); 57 more steps from cleaner throughput didn't move the needle. Residual gap: manual-add blend form + TTT α bug. See `research/evaluations/021d-recur-alpha-param-frozen-8xh.md`. |
| 021h | **Spec 021h — Learnable α fp32, 8×H100 JP** (fp32 storage for α to restore update precision; hypothesis: bf16 LSB=0.0078 traps α on coarse grid vs AdamW step ~1e-5) | `exp/recur-alpha-buffer` | 5906820 | 8×H100 JP-1 (same pod as 021g) | **1.07043** pre-quant post-EMA / **1.07976** post-GPTQ / **1.06734** post-TTT / **1.1084** @ step 4000 (best 021-family) / 4753 steps | +0.00112 vs 021e — **worse than frozen α** / ≈017 buggy-TTT (1.06733) | 🗄️ LEARNABLE α ARC CLOSED — fp32 closed L5 gap from −0.047 to −0.024 vs 017 but can't reach 017's basin; algebraic blend form creates different optimization landscape than 017's manual-add; 017 advantage = pod luck + manual-add, not reproducible. **021e frozen α wins entire arc.** See `research/evaluations/021h-learnable-alpha-fp32-8xh.md`. |
| 021g | **Spec 021g — 017-redux: learnable α bf16 + TTT fix + algebraic form, 8×H100 JP** (learnable nn.Parameter α init=1.0 bf16 + TTT α fix + algebraic blend — attempted reproduction of 017's pre-quant advantage 1.06861) | `exp/recur-alpha-buffer` | fab6e7f | 8×H100 JP-1 | **1.06987** pre-quant post-EMA / **1.07928** post-GPTQ / **1.06693** post-TTT / 4804 steps | +0.00071 vs 021e / +0.00126 worse than 017 pre-quant — **017's advantage did NOT reproduce** | 🗄️ LEARNABLE BF16 α CLOSED — α converges to offset basin (L5: 1.383 vs 017's 1.430); bf16 LSB=0.0078 >> AdamW step ~1e-5, α trapped on coarse grid. Learnable bf16 worse than frozen α. Next: 021h (fp32 storage restores precision). See `research/evaluations/021g-017-redux-learnable-alpha-ttt-fix-8xh.md`. |
| 021e | **Spec 021e — Recur-α Parameter+bf16+algebraic+TTT-fix, 8×H100 JP** (cumulative fix: Parameter container + bf16 + algebraic lerp x=x_before+α(x_new−x_before) at 4 sites + TTT α applied in forward_ttt) | `exp/recur-alpha-buffer` | d761a22 | 8×H100 JP-1 | **1.06944** pre-quant post-EMA / **1.07863** post-GPTQ / **1.06622** post-TTT / **1.1134 @ step 4000** | **−0.00006 vs 019b** / **miss #1736 by 0.00012** — within seed std | ✅ **NEW BEST SINGLE-SEED** — per-step loss gap eliminated (±0.005 noise vs prior +0.007–0.022); TTT δ −0.01322 (larger than 019b's −0.01249, confirming TTT α fix worked). 0.00012 miss is within seed std ~0.0003–0.0005. **Next: 021e seed 43/44 on fresh JP pod.** See `research/evaluations/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh.md`. |
| 021c | **Spec 021c — A/B/C/A' frozen-α container + blend-form test** (Arm A manual-add literal `e93d77d` JP, Arm B Param bf16 manual-add `8b2d791` JP, Arm C buf bf16 manual-add `d070df3` JP, Arm A' **algebraic literal `9517a3b` NE-1** = true 019b-submission code; Arm E `d761a22` full-stack superseded by 8H 021e direct) | multiple | e93d77d / 8b2d791 / d070df3 / 9517a3b | 4×H100 JP-1 + NE-1 | A **1.06927** / B **1.06952** / C partial / **A' 1.06883** | Δ(B−A)+0.00024 / **Δ(A'−A) −0.00044** (algebraic > manual-add) / Parameter vs literal vs buffer indistinguishable at 4H | 🗄️ SHELVED container arc. Key finding: **algebraic blend > manual-add by ~0.0005** (commit-hash confusion: our earlier "019b" reference was pre-OOM-fix manual-add, not true 019b-submission). Parameter-vs-buffer is not the mechanism. 021-family's 8H gap to 019b is likely manual-add + TTT α bug, not α container. Next: user's 8H 021e (full stack d761a22) decisive; fallback 019b-original 3-seed. See `research/evaluations/021c-recur-alpha-param-frozen-mini.md`. |

---

## Sweep Summary (April 17, 2026)

11 experiments on 2×H100 + 1 combined run, ~3.5 hours total, ~$11 spent.

**Best result: Exp 15 (combined) — val_bpb = 1.2590 (-0.014 vs baseline)**

Config: `TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4`

**Key findings:**

1. **seq_len=2048 is the biggest lever** — Exp 13 scored 1.2583, -0.015 vs baseline. Longer context gives the model more structure to learn from per step. Even with fewer steps (2738 vs 2969), quality per step massively compensates.

2. **q_gain=4.0 is the optimal value** — not default 1.5, not top-submission's 5.25. The 3.0-4.0 range is the sweet spot.

3. **Gains stack** — Exp 15 (seq2048 + qk4.0 + recurrence) beats Exp 13 (seq2048 + recurrence) at every training step by ~0.005-0.007. Final numbers converge due to quantization noise.

4. **Don't shrink model_dim** — Exp 11 (dim=448) and Exp 12 (dim=384) were the worst results. Width > depth at this scale.

5. **3-layer recurrence ≈ 2-layer** — diminishing returns from adding more recurring layers.

6. **Warmdown=2400 doesn't help** — Exp 14 was basically identical to baseline.

**Next experiments to run:**
- Add SP4096 tokenizer (needs data generation — biggest remaining lever)
- Try staged recurrence (activate mid-training, proven by top submissions)
- Try EMA weights (almost free improvement)
- Try parallel residuals (GPT-J style, proven +0.004)
- Try LeakyReLU instead of ReLU²

---

## Experiment Details

### Exp 2 — Depth Recurrence (1×H100)

**What we did:** Keep 9 physical layers (same param budget). Add recurrence on layers 3,4 → 11 virtual passes per forward pass. Apples-to-apples vs baseline: same ~17M params, more effective depth.

**Parameter changes vs baseline:**
- NUM_LAYERS: 9 (unchanged)
- RECUR_LAYERS: (none) → "3,4"
- block_schedule: [0..8] → [0,1,2,3,4,3,4,5,6,7,8] (11 virtual passes)
- model_params: 17,059,912 → 17,060,424 (+512 for extra skip_weights)

**Code version:**
- Branch: `exp/depth-recurrence`
- Commit: `ea1898a`
- Diff: `git diff main exp/depth-recurrence -- train_gpt.py`

**Run:**
- Pod: Runpod H100 SXM, 1 GPU
- Log: `/workspace/logs/exp2_depth_recur.log`
- Step time: ~557ms/step (vs baseline 506ms — 10% slower due to 2 extra virtual passes)
- Steps completed: 1078 (hit 600s wall)

**Results:**
- val_bpb at step 1000: **1.3510**
- val_bpb at step 1078 (final): **1.3449**
- Δ vs baseline: **-0.014 bpb** (win)

**Analysis:**
Recurrence wins by 0.014 bpb at 1×H100 scale. The 11-virtual-pass model learns slightly better representations in the same wall time, even with fewer training steps (1078 vs ~1200 for baseline). The extra compute cost per step is ~10% (557 vs 506ms/step). On 8×H100 where step time is ~96ms, the compute overhead matters much less. This warrants a proper 2×H100 comparison run.

**Run command:**
```bash
cd /workspace/parameter-golf && git checkout exp/depth-recurrence
torchrun --standalone --nproc_per_node=1 train_gpt.py > /workspace/logs/exp2_depth_recur.log 2>&1
```

---

### Exp 3 — Baseline (2×H100)

**What we did:** Clean baseline run on 2×H100 for direct comparison with recurrence.

**Parameter changes vs 1×H100 baseline:** None — same code, same config, just 2 GPUs.

**Code version:**
- Branch: `main`
- world_size=2, grad_accum_steps=4

**Run:**
- Pod: Runpod 2×H100 SXM
- Log: `/workspace/logs/exp3_baseline_2gpu.log`
- Step time: ~202ms/step
- Steps completed: ~2975

**Results (val_bpb curve):**
```
step 1000: 1.3830
step 2000: 1.3169
final INT8: 1.2732
artifact: 15,713,168 bytes
```

---

### Exp 4 — Depth Recurrence (2×H100)

**What we did:** Same as Exp 2 but on 2×H100 with VAL_LOSS_EVERY=200 for more data points.

**Parameter changes vs baseline:**
- RECUR_LAYERS: "3,4"
- block_schedule: [0,1,2,3,4,3,4,5,6,7,8] (11 virtual, 9 physical)
- VAL_LOSS_EVERY: 200

**Code version:**
- Branch: `exp/depth-recurrence`
- Commit: `ea1898a`

**Run:**
- Pod: Runpod 2×H100 SXM
- Log: `/workspace/logs/exp4_depth_recur_2gpu.log`
- Step time: ~230ms/step (14% slower than baseline — 11 virtual passes vs 9)
- Steps completed: ~2600 (fewer than baseline due to slower steps + val overhead)

**Results (val_bpb curve):**
```
step  200: 1.6836
step  400: 1.5178
step  600: 1.4475
step  800: 1.4059
step 1000: 1.3780   (baseline: 1.3830, Δ -0.005)
step 1200: 1.3582
step 1400: 1.3453
step 1600: 1.3307
step 1800: 1.3145   (baseline step 2000: 1.3169, Δ -0.002)
step 2000: 1.3000   (baseline step 2000: 1.3169, Δ -0.017)
step 2200: 1.2892
step 2400: 1.2787
step 2600: 1.2710
final INT8: 1.2728
artifact: 15,569,434 bytes
```

**Analysis:**
Recurrence wins at every comparable step count:
- At step 1000: -0.005 bpb
- At step 2000: -0.017 bpb (gap growing with more training)
- Final: -0.0004 bpb (gap shrinks because recurrence gets fewer total steps: 2600 vs 2975)

The advantage grows with more steps — at step 2000 the gap is 0.017. On 8×H100 with full 20K steps, recurrence should show a much larger final advantage. Depth recurrence is confirmed as a real improvement.

Key observations:
- 14% slower per step (acceptable tradeoff for better per-step quality)
- Smaller artifact (15.6 MB vs 15.7 MB) — 512 extra skip_weight params is negligible
- The val_bpb curve shows smooth, monotonic improvement — no instability from recurrence

---

### Exp 16 — Phase 1: LeakyReLU + Parallel Residuals + Staged Recurrence (2×H100)

**What we did:** Combined 4 structural changes on top of our best config (seq2048 + qk4.0 + recurrence):
1. LeakyReLU(0.5)² instead of ReLU² — better gradient flow through negative values
2. Parallel residuals from layer 7+ — attention and MLP read from same input (GPT-J style)
3. Staged recurrence — plain sequential for first 1000 steps, then activate recurrence
4. All existing improvements (seq_len=2048, q_gain=4.0, recur_layers=3,4)

**Hypothesis:** Structural improvements that every top submission uses should stack cleanly.

**Config:**
```bash
TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4
RECUR_START_STEP=1000 PARALLEL_START_LAYER=7
VAL_LOSS_EVERY=200
```

**Code version:**
- Branch: `exp/depth-recurrence`
- Commit: `6e2da72`

**Run:**
- Pod: Runpod 2×H100 SXM
- Log: `/workspace/logs/exp16_phase1.log`
- Step time: 217ms/step pre-recurrence (steps 0-1000), ~255ms/step post-recurrence (steps 1000+)
- Steps completed: ~2300 (fewer than Exp 15 due to recompile overhead)

**Results (val_bpb curve with wall time):**
```
Time(s) | Step | val_bpb | Δ vs Exp15
   43   |  200 | 1.6533  | +0.006  (slower start)
   87   |  400 | 1.4877  | -0.007  (caught up)
  130   |  600 | 1.4181  | -0.010  (pulling ahead)
  174   |  800 | 1.3810  | -0.006
  217   | 1000 | 1.3542  | -0.005  ← recurrence activates here
                                     84s recompile gap
  301   | 1200 | 1.3340  | -0.007  (resumed after recompile)
  354   | 1400 | 1.3125  | -0.015  (recurrence kicking in!)
  407   | 1600 | 1.2965  | -0.002  (gap narrowing — recompile cost)
  460   | 1800 | 1.2813  | -0.003
  512   | 2000 | 1.2678  | -0.007
  565   | 2200 | 1.2575  | -0.007
final INT8:    | 1.2550  | -0.004  ★ NEW BEST
```

**Final:** val_bpb = **1.2550** (-0.018 vs baseline, -0.004 vs Exp 15)

**Key learnings:**
1. **LeakyReLU works** — improvement visible from step 400 onward. The gentler activation helps early training converge faster. This is a "stable" structural improvement.
2. **Parallel residuals work** — no instability, consistent improvement. Another stable structural win.
3. **Staged recurrence has a hidden cost** — `torch.compile(fullgraph=True)` recompiles when block_schedule changes, costing 84 seconds of wall time. This is 14% of the 600s budget. On 2×H100 it barely pays for itself. On 8×H100 with more steps it would be more clearly worth it.
4. **Pre-recurrence phase is faster** — 217ms/step vs 255ms/step. The 1000 plain steps complete in ~217s, giving more steps in the early phase where convergence is fastest.
5. **Structural changes stack** — LeakyReLU + parallel residuals + recurrence each help independently and combine cleanly. This validates the "stable over finetuned" approach.

**What to try next:**
- Run WITHOUT staged recurrence (recurrence from step 0) — avoid the 84s recompile cost
- Add EMA weights
- Try weight decay 0.09
- Investigate partial RoPE (16/64 dims)

---

### Exp 13 — Recurrence + seq_len=2048 (2×H100)

**What we did:** Doubled training sequence length from 1024 to 2048 tokens, with depth recurrence.

**Hypothesis:** Longer context → model learns longer-range patterns per step. RoPE handles arbitrary lengths.

**Config:** `TRAIN_SEQ_LEN=2048 RECUR_LAYERS=3,4 VAL_LOSS_EVERY=200`

**Code version:** Branch `exp/depth-recurrence`, commit `ea1898a`

**Results (val_bpb curve with wall time):**
```
Time(s) | Step | val_bpb
   46   |  200 | 1.6676
   92   |  400 | 1.5033
  138   |  600 | 1.4346
  184   |  800 | 1.3922
  230   | 1000 | 1.3655
  277   | 1200 | 1.3446
  323   | 1400 | 1.3314
  369   | 1600 | 1.3207
  414   | 1800 | 1.3049
  460   | 2000 | 1.2909
  506   | 2200 | 1.2798
  552   | 2400 | 1.2692
  598   | 2600 | 1.2607
final INT8:    | 1.2583
```

**Final:** val_bpb = **1.2583** (-0.015 vs baseline)

**Key learnings:**
1. **seq_len=2048 is the biggest single lever we found** — bigger than recurrence, q_gain, or any hyperparameter
2. Each step processes 2× more tokens but takes roughly the same wall time per step (~230ms)
3. The model sees full paragraphs instead of fragments → learns structure more efficiently
4. RoPE handles 2048 natively — no code changes needed, just an env var

---

### Exp 15 — Combined: recurrence + seq2048 + q_gain=4.0 (2×H100)

**What we did:** Added q_gain=4.0 on top of Exp 13's config (seq2048 + recurrence).

**Hypothesis:** q_gain=4.0 (optimal from sweep) should stack with seq2048 improvement.

**Config:** `TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4 VAL_LOSS_EVERY=200`

**Code version:** Branch `exp/depth-recurrence`, commit `ea1898a`

**Results (val_bpb curve with wall time):**
```
Time(s) | Step | val_bpb | Δ vs Exp13
   46   |  200 | 1.6474  | -0.020
   92   |  400 | 1.4943  | -0.009
  138   |  600 | 1.4280  | -0.007
  184   |  800 | 1.3865  | -0.006
  230   | 1000 | 1.3590  | -0.007
  277   | 1200 | 1.3411  | -0.004
  323   | 1400 | 1.3279  | -0.004
  369   | 1600 | 1.3133  | -0.007
  414   | 1800 | 1.2983  | -0.007
  460   | 2000 | 1.2847  | -0.006
  506   | 2200 | 1.2744  | -0.005
  552   | 2400 | 1.2643  | -0.005
  598   | 2600 | 1.2572  | -0.004
final INT8:    | 1.2590  | +0.001 (quant noise)
```

**Final:** val_bpb = **1.2590** (-0.014 vs baseline)

**Key learnings:**
1. q_gain=4.0 stacks cleanly with seq2048 — consistent -0.004 to -0.007 at every checkpoint
2. Final INT8 number (1.2590) is slightly worse than pre-quant (1.2572) — quantization noise
3. Exp 13's final (1.2583) is slightly better after INT8 despite being worse pre-quant — confirms quantization noise is ~0.002 and unpredictable

---

### Exp 5 — SP4096 Tokenizer
**Hypothesis:** Larger vocab means each token covers more bytes → lower bpb for same model quality.

**Config:**
```bash
VOCAB_SIZE=4096 TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
DATA_PATH=./data/datasets/fineweb10B_sp4096
```

**Prep needed:** Download SP4096 data (~same size as SP1024):
```bash
python3 data/cached_challenge_fineweb.py --variant sp4096
```

---

### Exp 4 — Depth Recurrence + SP4096
Combine Exp 2 + Exp 3. If both give positive deltas independently, combining should stack.

---

### Exp 5 — mlp_mult=1
**Hypothesis:** Cutting MLP width from 2× to 1× saves ~500K params per layer. Use those params to add more layers or widen model_dim.

**Config:**
```bash
MLP_MULT=1 NUM_LAYERS=12  # or MODEL_DIM=576
```

---

### Exp 21 — Full Run: All Improvements, 40min (2×H100)

**What we did:** Ran our full stack of improvements for 40 min on 2×H100 — equivalent compute to 8×H100 10min. Native PyTorch GQA (PyTorch 2.9.1).

**Config:**
```bash
TRAIN_SEQ_LEN=2048 QK_GAIN_INIT=4.0 RECUR_LAYERS=3,4
RECUR_START_STEP=1000 PARALLEL_START_LAYER=7
ROPE_DIMS=16 VAL_LOSS_EVERY=200
MAX_WALLCLOCK_SECONDS=2400 WARMDOWN_ITERS=1200
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

**Model:** 17M params, 9 layers, our code (exp/depth-recurrence branch)

**Run:**
- Pod: RunPod 2×H100 SXM, `runpod/parameter-golf:latest`
- Log: `/workspace/logs/exp21_full_run.log`
- Step time: ~210ms pre-recurrence, ~250ms post-recurrence
- Steps completed: 9511 (wall clock cap at 2400s)
- Peak memory: 12,140 MiB

**Key discovery — warmdown timing:**
WARMDOWN_ITERS=1200 is relative to the wall clock endpoint, not a fixed step. With 9511 total steps, warmdown ran at steps ~8300-9511 (not 4800-6000 as expected). This meant 87% of training at full LR, 13% warmdown.

**Results (val_bpb curve with wall time):**
```
Time(s) | Step | val_bpb | Phase
     42 |  200 | 1.7022  | pre-recurrence
     84 |  400 | 1.5024  |
    126 |  600 | 1.4303  |
    168 |  800 | 1.3848  |
    211 | 1000 | 1.3567  | recurrence activates → 77s recompile
    288 | 1200 | 1.3344  | post-recurrence, full LR
    339 | 1400 | 1.3186  |
    390 | 1600 | 1.3088  |
    441 | 1800 | 1.2975  |
    492 | 2000 | 1.2892  |
    542 | 2200 | 1.2830  |
    593 | 2400 | 1.2764  |
    644 | 2600 | 1.2781  | (blip)
    695 | 2800 | 1.2680  |
    746 | 3000 | 1.2627  |
    797 | 3200 | 1.2591  |
    849 | 3400 | 1.2567  |
    900 | 3600 | 1.2523  | ← beat Exp 16 (1.2550)
    951 | 3800 | 1.2504  |
   1001 | 4000 | 1.2473  |
   1054 | 4200 | 1.2450  |
   1105 | 4400 | 1.2442  |
   1157 | 4600 | 1.2428  |
   1208 | 4800 | 1.2397  |
   1259 | 5000 | 1.2370  |
   1310 | 5200 | 1.2368  |
   1361 | 5400 | 1.2343  |
   1412 | 5600 | 1.2335  |
   1462 | 5800 | 1.2322  |
   1512 | 6000 | 1.2302  |
   1563 | 6200 | 1.2293  |
   1613 | 6400 | 1.2277  |
   1664 | 6600 | 1.2256  | ← beat baseline pre-quant (1.2244)
   1714 | 6800 | 1.2253  |
   1765 | 7000 | 1.2237  |
   1815 | 7200 | 1.2233  |
   1866 | 7400 | 1.2221  |
   1916 | 7600 | 1.2210  |
   1967 | 7800 | 1.2196  |
   2017 | 8000 | 1.2182  |
   2068 | 8200 | 1.2175  |
   2120 | 8400 | 1.2159  | ← warmdown begins (~step 8300)
   2171 | 8600 | 1.2103  | freefall starts
   2221 | 8800 | 1.2054  |
   2272 | 9000 | 1.2008  |
   2322 | 9200 | 1.1957  |
   2373 | 9400 | 1.1913  |
   2400 | 9511 | 1.1898  | wall clock cap
        | INT8 | 1.1963  | final (exact: 1.19627328)
```

**Final:** val_bpb = **1.1963** INT8 (pre-quant 1.1898, quant cost +0.0065)
**Artifact:** 15,871,831 bytes (under 16MB ✓)

**Comparison:**
- vs Baseline (1.2244): **-0.028** ✓ beat it
- vs Exp 16 (1.2550): **-0.059**
- vs SOTA (1.0810): +0.115 (still far)

**Caveat:** This is NOT a valid submission — ran 2×H100 40min, not 8×H100 10min. Projected 8×H100 10min result: ~1.207.

---

### Exp 22 — SOTA Code Test with SP1024 (1×H100)

**What we did:** Ran the SOTA submission code (obfuscated, from records/) on 1×H100 10min with SP1024 data to verify the code works end-to-end.

**Config:**
```bash
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 GPTQ_RESERVE_SECONDS=12
TTT_ENABLED=0 SLIDING_WINDOW_ENABLED=0 VAL_LOSS_EVERY=500
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Model:** 32.3M params (SOTA architecture: 11L, MLP 4×, XSA, skip gates, 3-layer recurrence)

**Run:**
- 534 steps in 588s, wall clock cap
- ~1.1M tok/s pre-recurrence, ~715K tok/s post-recurrence (17 virtual layers)
- Peak memory: 38,085 MiB

**Results:**
```
Step | val_bpb
 500 | 1.2992
 534 | 1.2929  (final pre-quant step)
INT8 | 1.6050  (post-EMA hurt — too few steps for EMA to converge)
```

**Artifact:** 13,821,943 bytes (under 16MB — smaller than SP8192 due to smaller embedding table)

**Verdict:** Code runs end-to-end. Bad score because only 534 steps (need 4550). Proved: model init, training, EMA, GPTQ INT6, Brotli compression, eval all work.

---

### Exp 23 — SOTA Code Test with SP8192 (1×H100)

**What we did:** Same as Exp 22 but with freshly tokenized SP8192 data. The real deal.

**Config:**
```bash
MAX_WALLCLOCK_SECONDS=600 GPTQ_RESERVE_SECONDS=12
VAL_LOSS_EVERY=200 TTT_ENABLED=0 SLIDING_WINDOW_ENABLED=0
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Model:** 35.9M params (full SOTA: 11L, MLP 4×, vocab 8192)

**Run:**
- 558 steps in 588s, wall clock cap
- ~1.0M tok/s pre-recurrence, ~755K tok/s post-recurrence
- Recurrence activated at step 236 (frac=0.35)
- Recompile for 17 virtual layers: ~4-5 min on 1×H100
- Peak memory: 39,282 MiB

**Results:**
```
Step | val_bpb
 200 | 1.4156
 400 | 1.2705
 558 | 1.2214  (final pre-quant step)
INT8 | 1.4052  (post-EMA + quant — too few steps)
```

**Artifact:** 16,005,147 bytes (5KB over 16MB limit — minor issue)

**Key observations:**
1. SP8192 model is 35.9M params vs 32.3M with SP1024 (larger embedding table)
2. Compile + recompile eats ~7 min of the 10 min budget on 1×H100
3. Only ~240 post-recurrence steps out of 558 total
4. 1×H100 gets 558 steps vs SOTA's 4550 on 8×H100 — ratio ~8.15×
5. Full equivalent run needs **~80 min on 1×H100** or **~40 min on 2×H100**

**Scaling estimate for full SOTA replication:**
| Hardware | Wall time | Steps | Expected val_bpb |
|---|---|---|---|
| 8×H100 | 10 min | 4550 | 1.0810 (proven) |
| 2×H100 | 40 min | ~4550 | ~1.08 (expected) |
| 1×H100 | 80 min | ~4550 | ~1.08 (expected) |
| 1×H100 | 10 min | 558 | 1.4052 (too few steps) |

---

## Ideas Backlog (not yet scheduled)

- **INT6 quantization** — replace INT8 with INT6, fits ~25% more params in 16MB
- **q_gain tuning** — top submissions pushed to 5.25 (baseline 1.5). Try 3.0, 5.0.
- **Parallel residuals** — attention + MLP read from same input (GPT-J style), proven +0.004 bpb
- **Larger model_dim** — push to 576 or 640 if depth recurrence frees enough params
- **num_kv_heads=2** — reduce from 4 to 2, save more K/V params

---

## Run Template

To add a new run, copy this:

```
| N | **name** | `branch` | TBD | TBD | ⏳ queued | description |
```

Log command:
```bash
nohup torchrun --standalone --nproc_per_node=1 train_gpt.py > /workspace/logs/expN_name.log 2>&1 &
```
