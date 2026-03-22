# Parameter Golf Build Narrative — How We Got Here

## Team & Tools
- **Builder:** Anthony Maio (anthony-maio) + Claude Opus 4.6 (1M context)
- **Kernel generation:** Makora (unlimited beta credits)
- **Strategic advisors:** Multi-model council (GPT-5.4, Gemini 3.1 Pro, Claude Sonnet 4.6, Sonar, Nemotron 3 Super)
- **Compute:** RunPod 8xH100 SXM ($21.52/hr), 1xH100, Colab, local 3090s
- **Competition:** OpenAI Parameter Golf — best LM in 16MB, 10 min training on 8xH100

## Timeline & Evolution

### Day 1 (March 20) — Forking & First Approaches

**Started** by forking openai/parameter-golf, setting up RunPod CLI, installing Makora skills.

**Phase 1: TTT + SOTA Graft (PR #175)**
First attempt was mechanical — graft LoRA test-time training onto the current SOTA training recipe. The idea: two orthogonal improvements (training quality + eval adaptation) that nobody had combined. Built and pushed within hours.

**Phase 2: Depth Recurrence**
Hypothesis: shared weights looped N times = "free" effective depth under 16MB cap. Built 5 unique blocks × 4 loops = 20 effective layers at dim=640. The council loved it theoretically.

Reality: depth recurrence costs 2.7x per step. Got 4,000 steps instead of SOTA's 7,300. Result: 1.2613 bpb. The council unanimously said **abandon it** — the competition is throughput-bound, not parameter-bound. They were right.

**Phase 3: Kitchen Sink**
Integrated every technique from the leaderboard (MLP 3x, SmearGate, BigramHash, int6+zstd, SWA, OrthoInit) into depth recurrence. Validated full pipeline on Colab. Then ran on 8xH100 — 1.2613 with recurrence, 1.2015 without (standard 9L). Recurrence was the wrong bet.

### The TTT Debugging Marathon (8+ hours)

**The bug:** LoRA TTT made results WORSE on our model. Spent 8+ hours systematically eliminating hypotheses:

1. ❌ torch.compile — tested COMPILE=0, same result
2. ❌ SWA — tested SWA=0, same result
3. ❌ Int6 quantization — tested pre-quant model, same result
4. ❌ Learning rate too high — tested lr=0.001, even worse
5. ❌ SmearGate — tested with minimal model, not the cause
6. ❌ BigramHash — tested BIGRAM_VOCAB_SIZE=0, same result
7. ✅ **Cross-test revealed:** fresh uncompiled model produces catastrophic TTT (1.797 bpb) while compiled base_model works (1.306). torch.compile's graph IS required for LoRA TTT.
8. But even passing compiled base_model, TTT still failed on our enhanced architecture (SmearGate + BigramHash + MLP 3x + OrthoInit).

**Resolution:** The model council pointed out the REAL SOTA (1.1303, FarnsworthEngine) uses **full-weight SGD TTT** instead of LoRA TTT. Completely different approach that bypasses all the LoRA/compile issues.

### Day 2 (March 21) — Matching SOTA

**Key insight from council:** Our 1.2015 vs SOTA's 1.1483 gap was primarily **hyperparameters** (seq1024 vs seq2048, matrix_lr=0.04 vs 0.02), not architecture.

**Took PR #162's exact script** (proven 1.1483), grafted full-weight SGD TTT onto it, updated hyperparams to match FarnsworthEngine (11L, NTK-RoPE 50k, WD=0.04).

**Pod lottery:** Spun 3 pods, benchmarked, kept fastest (105ms/step base → 123ms with 11L).

**Results so far:**
- Sliding window without TTT: **1.1434 bpb** (beats old SOTA!)
- TTT adapts successfully (3 epochs, loss decreasing)
- Crashed twice on RunPod spot instances; switched to SECURE
- Hit a variable scoping bug in TTT eval (fixed)
- Final run with TTT in progress

### Makora Custom Kernels (parallel track)

**8 kernel jobs submitted** across Triton and CUDA:

| Kernel | Best Speedup |
|--------|-------------|
| Fused RMSNorm+QKV | 1.47x |
| Fused ReLU² MLP | 1.23x |
| Fused softcap+CE | 1.21x |
| Fused TTT MLP step | 1.21x |
| Fused resid_mix+RMSNorm | 1.08x |
| Fused Q/K RMSNorm+RoPE+qgain | generating |

First attempt at integrating Makora kernels failed (alignment bugs, incorrect results). Root cause: Makora validates with single forward pass but integration context involves autograd, autocast, DDP, iterative application. Filed detailed feedback to Makora team.

Hand-wrote a fused RMSNorm+linear kernel using our Triton skills — 1.32x speedup, correct output. But only 0.2% per-step impact at H100 speeds (the ops are already fast).

**The kernel opportunity** is compounding: 1.47x on RMSNorm+QKV + 1.23x on ReLU² MLP + others, each called 11-22x per step across 11 layers. No other competitor has custom kernels.

## Key Decisions & Lessons

1. **Depth recurrence was wrong for this competition.** Trades compute for params, but competition is compute-bound. The council saved us from wasting more time.

2. **Hyperparameters > architecture innovation** at this scale. seq2048 vs seq1024 mattered more than any architectural choice.

3. **torch.compile is load-bearing** for TTT — creating fresh uncompiled models produces silently wrong results. CastedLinear's fp32/bf16 interplay behaves differently under compile vs eager.

4. **Full-weight SGD TTT > LoRA TTT** on enhanced architectures. Simpler, more robust, works with SmearGate/BigramHash.

5. **Model councils are extremely valuable** for strategic decisions. The multi-model consensus on abandoning recurrence and the LR/TTT debugging were pivotal.

6. **RunPod spot instances are unreliable** for long runs. Use SECURE cloud for competition submissions.

7. **Custom kernels are the endgame.** Nobody else has them. The long game (April 30 deadline) favors this unique advantage.

## Current State

- Best result: ~1.14 bpb sliding window (TTT pending)
- 8 Makora kernel jobs generating
- Full-weight SGD TTT validated (epochs complete, eval bug fixed)
- PR ready for submission
- Compute grant application drafted
