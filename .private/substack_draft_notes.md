# Parameter Golf: An AI-Human Partnership Story
## Draft Notes for Substack Article

### The Hook
A ML engineer and an AI pair-programmer entered OpenAI's Parameter Golf competition — a challenge to train the best language model that fits in 16MB and trains in 10 minutes on 8xH100s. Over 5 days of intense collaboration, they went from zero to matching the verified SOTA, debugging impossible bugs, building custom GPU kernels, and orchestrating a "model forum" of 5 frontier LLMs to guide strategy. This is the story of what worked, what didn't, and what it's like to compete at the frontier of ML optimization with an AI partner.

---

### Timeline & Evolution

#### Day 1 (March 20): The Grand Plan That Was Wrong
- **Starting point**: Zero code, zero infrastructure. Competition 2 days old, SOTA at 1.1748 bpb.
- **Initial strategy**: Depth recurrence (5 unique blocks × 4 loops = 20 effective layers) + custom Makora GPU kernels + LoRA test-time training (TTT). The theory was elegant — shared weights give "free" depth under the 16MB cap, and automated kernel generation via Makora would be our unfair advantage.
- **First run on Colab B200**: Training worked! 15.6M params, 20 effective layers, loss dropping nicely at 357ms/step.
- **First 8xH100 run**: 1.2956 bpb at 153ms/step. Promising but 0.15 bpb behind SOTA.
- **TTT debugging begins**: LoRA TTT made results WORSE on our architecture. Spent hours investigating.
- **Key lesson**: Built 8 custom Makora kernels (fused RMSNorm+QKV at 1.47x, fused ReLU² MLP at 1.26x, fused softcap+CE at 1.75x). All passed validation. None helped training because torch.compile already optimized those paths on H100.

#### Day 2 (March 21): The Model Council Changes Everything
- **The pivot**: we Convened a "model council" — GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro, Sonar, and Nemotron Super — to evaluate our depth recurrence strategy.
- **Unanimous verdict**: "Abandon depth recurrence. It trades compute for parameters, but you're compute-bound, not parameter-bound." The math was brutal: 2.7x slower per step = 45% fewer training steps = 60% less data seen.
- **TTT root cause found**: Through systematic A/B testing across 8+ configurations, discovered the bug was NOT quantization, NOT learning rate, NOT SmearGate — it was the interaction between torch.compile and our modified architecture. The original TTT code worked on the original architecture; our additions (SmearGate + BigramHash) broke the compiled model's TTT path.
- **New direction**: Switched to standard 9-layer architecture matching PR #162's proven recipe. First run: 1.2015 bpb. Then with FA3 import fix: 1.1574 bpb. Getting close.
- **First PR submitted**: PR #376 at 1.1401 bpb — #1 on the merged leaderboard at the time.

#### Day 3 (March 22): The Next-Gen Stack
- **Intelligence gathering**: Analyzed every top PR on the leaderboard. Discovered the competition had converged on a specific stack: 11 layers, XSA, Partial RoPE, LN Scale, VE128, EMA, Late QAT, GPTQ-lite.
- **Strategic feedback from Sonnet 4.6**: "Your current submission is technically competent but strategically off-target. The best move is not 'improve #376.' It is 'replace #376 as the optimization target with a #414-class base.'" Brutal but correct.
- **Custom kernel autograd debugging**: Built autograd-compatible Triton kernels (FusedResidMixRMSNorm, FusedReLU2MLP). An agent found two critical bugs: (1) .to(dtype) detaches from autograd graph, (2) persistent kernel grid capped below actual tile count. Fixed both — but they still added 38ms/step overhead vs torch.compile. Kernels shelved.
- **Closed PR #376**: Accepted the council's recommendation. Started fresh branch `submission/reproduce-414`.

#### Day 4 (March 23): The FA3 Saga
- **The missing piece identified**: FlashAttention-3 Hopper kernels. Our runs used FA2 or SDPA fallback (99-131ms/step). PR #414 used FA3 at 84ms/step. The entire speed gap was one CUDA extension.
- **Build hell**: FA3 Hopper requires building ~60,000+ CUDA kernels from the Dao-AILab repo. First attempt: 20GB root disk filled up. Second attempt: moved to /workspace but pip used /tmp. Third attempt: 100GB root disk, but pod was in India with flaky network. Fourth attempt: ninja couldn't install due to network timeout.
- **Finally succeeded**: After migrating the pod to a new region, the build completed. 442MB wheel, 61,300 lines of build log.
- **The result**: 1.1229 bpb at 88ms/step with FA3. Matched the merged SOTA on first try.
- **But**: Artifact was 16.16MB — 157KB over the 16MB limit.

#### Day 5 (March 24): Closing the Gap — and Beating SOTA
- **Council identifies lzma**: The competing PR #549 uses lzma compression, not zstd. lzma is stdlib (no pip install needed!) and compresses 2-5% tighter. This is how they fit MLP 3x + BigramHash 3072 into 15.95MB.
- **The MLP 2.875x detour**: First attempt to fit under 16MB by shrinking MLP from 3x to 2.875x (tensor-core aligned at hidden=1472, caught by Gemini). Cost ~0.002 bpb. Ran 3 seeds — seed 42 still overflowed at 16.7MB due to seed-dependent weight entropy under zstd compression.
- **Implemented council's full recommendation**: LeakyReLU(0.5)² (one-line activation change worth -0.002 bpb), Value Residual Learning (VRL, ~20 lines, sigmoid-gated layer-0 V residual), restored MLP to 3.0x, switched to lzma compression.
- **3-seed results**: 1.1234 / 1.1225 / 1.1228 = **mean 1.1229 (std 0.0005)**. All artifacts under 16MB. All valid.
- **PR #657 submitted**: Beats the merged SOTA (PR #414's 3-seed mean of 1.1233) by 0.0004.
- **The gap to the frontier**: PR #549 claims 1.1194 with legal TTT. Our pre-TTT base (1.1229) is stronger than theirs (1.1218). Adding TTT should push us to ~1.120 or better.

---

### What Worked

1. **The Model Council** — Using 5 frontier LLMs as strategic advisors was the single highest-ROI decision. They correctly identified depth recurrence as net-negative (saving us a week of wasted compute), diagnosed the TTT bug mechanism, found the lzma compression insight, and provided ablation-backed technique rankings. The models disagreed on specifics (kernel priority, TTT timing, quantization strategy) but converged on the big calls.

2. **Systematic debugging** — The TTT bug took 8+ hours to diagnose but the methodology was sound: systematic A/B tests isolating one variable at a time (torch.compile, SWA, SmearGate, BigramHash, quantization, learning rate). Each test eliminated a hypothesis until only the real cause remained.

3. **Intelligence gathering** — Reading every top PR's README, ablation tables, and code was essential. The competition is fundamentally about information — knowing which techniques stack and which don't. PR #549's ablation table literally showed us the exact bpb value of each technique.

4. **Rapid iteration** — The ability to spin up 8xH100 pods on RunPod, run a 10-minute experiment, and get results was critical. We ran 30+ full training runs in 5 days.

### What Didn't Work

1. **Depth recurrence** — Elegant theory, wrong regime. At 5M parameters on 8xH100, you're compute-bound not parameter-bound. The 2.7x per-step overhead destroyed any benefit from effective depth.

2. **Custom GPU kernels for training** — 8 Makora-generated kernels, all passing validation benchmarks (1.2-1.75x speedup), but torch.compile on H100 already optimizes the same operations. Net effect: +38ms/step overhead from the autograd wrapper. The one kernel path that might have worked (fused linear+CE for the loss head) was never fully explored.

3. **LoRA TTT** — Broke on every architecture we tried it on. The interaction between compiled models, SmearGate, BigramHash, and per-document adaptation created an impossible optimization landscape for rank-8 LoRA adapters.

4. **Packed int6 binary format** — Implemented custom 6-bit packing (3 bytes per 4 values). Didn't help — the bottleneck was weight entropy under compression, not storage overhead.

5. **Building FA3 from source on RunPod** — Cost ~$100+ in GPU time across 4+ failed attempts. The root disk size, missing packages, and network issues on different regions made this absurdly difficult. The competition should pre-install FA3 on the official template.

### What We Learned About AI-Human Collaboration

1. **AI excels at synthesis, humans excel at judgment** — The model council could analyze 6 PRs, 10 techniques, and produce ranked recommendations in minutes. But the human had to decide "trust this, ignore that" — especially when models disagreed.

2. **Different AI models have different blind spots** — GPT-5.4 refused to help (OpenAI competition conflict), Nemotron gave confidently wrong architecture advice, but Gemini caught the tensor-core alignment issue (MLP 2.875x = 1472 = 32×46) that nobody else noticed.

3. **AI as infrastructure operator is powerful but fragile** — SSH into pods, launch training, poll for results, stop pods — all automated through Claude Code. But SSH timeouts, port changes, and pod restarts created constant error handling. The "happy path" automation was 10 lines; the error recovery was 100.

4. **The "overnight run" pattern works** — Setting up detached training with polling and auto-shutdown let the human sleep while experiments ran. Results waiting in the morning.

5. **Memory across sessions is critical** — The competition spanned 5 days across many conversation sessions. Project memory (what we tried, what worked, key findings) was essential for maintaining continuity.

---

### Technical Deep-Dive Sidebar Ideas

- **The anatomy of a 16MB language model**: What fits in 16MB and why every byte matters
- **FlashAttention 3 vs 2 vs SDPA**: Real benchmarks from our runs (242ms → 131ms → 88ms)
- **Why torch.compile beats custom Triton kernels at small scale**: Our kernel benchmarks and why the speedrun community's instinct was right
- **The model council methodology**: How to use multiple LLMs as strategic advisors (prompt design, synthesis, disagreement resolution)
- **Legal TTT: The eval-time compute arms race**: Score-first protocols, the memorization debate, and where the rules break down

---

### Key Numbers

| Milestone | BPB | Date | Key Change |
|-----------|-----|------|------------|
| Baseline | 1.2244 | Mar 18 | OpenAI starting point |
| Our first run (depth recurrence) | 1.2956 | Mar 20 | 5×4 recurrence, 8xH100 |
| After abandoning recurrence | 1.2015 | Mar 21 | Standard 9L, SOTA stack |
| First PR submitted | 1.1401 | Mar 21 | 11L, int5, full stack |
| With FA3 Hopper (MLP 3.0x, over 16MB) | 1.1229 | Mar 24 | True Hopper attention kernels |
| With lzma + LeakyReLU² + VRL (VALID) | **1.1229** | Mar 24 | Full competitive stack, 3-seed mean |
| Competition merged SOTA (PR #414) | 1.1233 | Mar 23 | 3-seed mean, verified |
| Frontier claim (PR #549) | 1.1194 | Mar 23 | + legal TTT |
| **Our PR #657** | **1.1229** | **Mar 24** | **Beats merged SOTA** |

### Credits & Acknowledgments
- OpenAI for hosting the competition and $200 in compute credits
- RunPod for GPU infrastructure
- Makora for automated kernel generation (beta access)
- The model council: Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro, Sonar, Nemotron Super
- The open-source NanoGPT speedrun community whose techniques form the foundation
- Specific PRs credited: #414 (signalrush), #549 (abaybektursun), #461 (Christopher-Lee-McClendon), #493 (parinzee), #399 (abaybektursun)
