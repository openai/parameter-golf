# Parameter Golf — Differentiation Brainstorm (v0.1)

**Date:** 2026-04-14
**Author:** sscswapnil + Claude
**Phase:** Planning-only (Apr 14–20). No code changes until Apr 21.
**Goal:** Beat SOTA (~1.1183 BPB, Apr 2026) with a *differentiated* approach that other teams are unlikely to converge on in 2 weeks.

---

## 1. Hard constraints distilled from the rules

| Constraint | Number | Implication for strategy |
|---|---|---|
| Artifact size | **16,000,000 bytes decimal** (code + compressed weights) | ~8–10MB weights realistic after code; every saved byte = more capacity |
| Train time | **10 min on 8×H100** | No multi-epoch regimes; no huge teacher→distill pipelines inside train |
| Eval time | **10 min on 8×H100** | TTT / iterative refinement has a budget; can't be unlimited |
| Network | **No network at eval** | All artifacts self-contained |
| TTT | Legal **only on already-graded val tokens** | Causal/streaming; no peeking |
| Tokenizer changes | Allowed but heavy scrutiny | High risk/reward |
| Libraries | Free (no code-byte cost) | Offload weight-decoding logic to a lib if possible |
| Stat sig | **0.005 nats @ p<0.01**, usually 3-seed avg | One-shot lucky seeds don't count |

**Single biggest leverage point:** The 10-minute training cap. SOTA is converging because everyone is running the same recipe inside this cap. Anything that gets *more signal per training second* is a moat.

---

## 2. The six candidate angles

For each: **Hypothesis → Expected gain → Rule-legality → Complexity → Go/No-go criteria → Risk.**

### Angle A — Weight-distribution-matched entropy coder (ANS / arithmetic coding)

**Hypothesis.** After QAT with clipped int6, per-row weights are approximately Laplacian/Gaussian with known per-row scale. A generic compressor (zstd) doesn't exploit this. An arithmetic or rANS coder parameterized by the empirical per-row distribution gets near-entropy compression. Expected **15–30% size reduction** vs zstd-22 on the same int6 weights.

**Expected gain.** 6.0MB → ~4.5MB → reinvest 1.5MB into capacity (extra layer, bigger MLP, or larger vocab). Capacity→BPB mapping is roughly 1MB ≈ 0.002–0.004 BPB at this scale, so a realistic win of **0.005–0.010 BPB**.

**Rule-legality.** ✅ Fine. Compression logic lives in `train_gpt.py` (eats code bytes) but the trade is positive: a few hundred bytes of coder code for MB of weight savings. Libraries (e.g., `constriction`, `torchac`) can do the heavy lifting for free.

**Complexity.** Medium. Need to (1) fit per-row distribution parameters during packing, (2) encode with rANS, (3) decode at model-load time, (4) ensure determinism. ~1 day of work.

**Go/No-go.**
- **Go if:** offline experiment shows ≥15% size reduction vs zstd on the v1 quantized weights.
- **No-go if:** <10% reduction (marginal, not worth the code bytes and complexity).

**Risk.** Low. Worst case we revert to zstd. Can be added as a drop-in replacement for the compression step.

**Priority: HIGH (first bet to prototype).**

---

### Angle B — Learned vector / product quantization (VQ / PQ)

**Hypothesis.** Uniform int6 is optimal only if the weight distribution is uniform. It isn't. Learned codebooks (k-means over sub-vectors of size 4–8) can match int8 accuracy at an **effective 3–4 bits/weight**, i.e. ~40% smaller than int6.

**Expected gain.** 6.0MB → ~3.8MB → reinvest 2.2MB into capacity. Potential **0.010–0.015 BPB** if QAT can be adapted to VQ codebooks (QAT-VQ is a known technique).

**Rule-legality.** ✅ Fine. Codebooks count as weights, decoder is code.

**Complexity.** HIGH. Need (1) QAT pass that gradients through codebook assignment (Gumbel-softmax or STE on nearest-code), (2) codebook learning schedule, (3) decode-time lookup. 2–3 days of work with non-trivial debugging. Risk of training instability.

**Go/No-go.**
- **Go if:** a 1-day sanity prototype on CPU using pretrained v1 weights shows PQ (no retrain) can hit int4-equivalent RMS error with a codebook overhead <5%.
- **No-go if:** training instability in QAT-VQ causes >0.02 BPB regression vs int6 in the first seed.

**Risk.** Medium-high. QAT-VQ is known-hard. Fallback: use PQ as *post-training* quantization (no QAT) — still ~20% size win but smaller BPB headroom.

**Priority: MEDIUM-HIGH (prototype after A).**

---

### Angle C — Data curation & curriculum (compute-free BPB wins)

**Hypothesis.** Every team uses the same FineWeb shards in the same order. Smarter data *selection* within the 10-minute train cap gives more signal per step. Specifically:
1. **Deduplication** within shards (MinHash/SimHash over 8-grams) — removes redundant gradients.
2. **Loss-based curriculum** — use a tiny pilot model's per-sample loss to re-order shards so hard examples come late.
3. **Length curriculum** — start short (512), end long (2048); reduces wasted attention on padding.

**Expected gain.** Published results show 5–15% sample-efficiency gain from good curation on small LMs. At our scale: **0.003–0.010 BPB**, stackable with every other angle.

**Rule-legality.** ✅ Fine per the FAQ (data dedup/reordering is offline prep, not external compute).
- ⚠️ BUT: if the competition mandates a specific data ordering (check `data/cached_challenge_fineweb.py`), reordering may be disallowed. **Must verify.**

**Complexity.** Low-medium. Dedup is 3 hours with `datasketch`. Curriculum sorting is an offline preprocessing pass. The training loop barely changes — just consumes a reshuffled shard.

**Go/No-go.**
- **Go if:** rules allow custom data ordering AND dedup removes ≥5% of tokens.
- **No-go if:** rules pin the data order/shuffle seed.

**Risk.** Low for dedup; medium for curriculum (can overfit to the pilot model's mistakes).

**Priority: HIGH (check legality first — if allowed, this is nearly free).**

---

### Angle D — Hybrid transformer + SSM (Mamba-2) layers

**Hypothesis.** Mamba-2 layers have subquadratic compute and strong per-parameter efficiency at short contexts. Replace 2–3 transformer layers with Mamba-2 blocks; gain compute budget to add capacity elsewhere OR process longer sequences.

**Expected gain.** Uncertain at this scale. Published SSM/transformer hybrids show ~2–5% perplexity win at matched params. At 16MB budget: **0.002–0.008 BPB**, but with high variance.

**Rule-legality.** ✅ Fine. SSMs are just another layer type.

**Complexity.** HIGH. Mamba-2 requires careful init + the `mamba-ssm` package + CUDA kernels. Debugging mixed-layer gradients eats time. New quantization story — Mamba's SSM state matrices don't quantize as cleanly as linear layers.

**Go/No-go.**
- **Go if:** a 1-shard sanity run of a 2-layer Mamba-only tiny model reaches a reasonable BPB (confidence that the ssm layer actually works in our setup).
- **No-go if:** integration eats >1 day (too risky with Apr 30 deadline).

**Risk.** HIGH — integration cost is real; quantization compatibility is an open question.

**Priority: LOW (exciting but time-expensive; only revisit if A+C ship early).**

---

### Angle E — Tokenizer rethinking

**Hypothesis.** The leaderboard uses 1024 BPE baseline and 8192 BPE in some SOTA entries. A **smaller vocab** frees embedding parameters for the transformer body. A **better-matched vocab** (e.g. morphological BPE, byte-level with learned n-gram merges) could lower BPB by better compressing the actual text distribution.

**Expected gain.** Uncertain; tokenizer changes are scrutinized and can mask real gains with evaluation artifacts. Published results vary wildly. **0.003–0.015 BPB** upside; **0.020+ BPB regression** downside if done wrong.

**Rule-legality.** ⚠️ Allowed but heavy scrutiny. Risk of rejection if BPB calculation isn't airtight across vocabs. The eval metric is bits-per-*byte*, so changing vocab changes the number of tokens but not the denominator — should be fair by construction, but any bug looks like cheating.

**Complexity.** MEDIUM. Training a new BPE is easy (`sentencepiece`). Integrating it, recomputing the BPE-to-byte loss conversion, and proving correctness to reviewers is harder.

**Go/No-go.**
- **Go if:** smaller vocab (e.g. 512) in a 1-shard test shows no BPB regression vs 1024, AND reviewer-proof BPB conversion is implemented.
- **No-go if:** rejection risk is ambiguous. Deadline is too tight to fight a review.

**Risk.** MEDIUM (implementation) + HIGH (review). Not the bet to make with 16 days left.

**Priority: LOW-MEDIUM (skip for this round; park for post-competition).**

---

### Angle F — Cross-layer weight sharing + per-layer low-rank deltas

**Hypothesis.** ALBERT-style: tie `{Q,K,V,O,MLP}` weights across groups of layers. Recover lost capacity via per-layer **low-rank deltas** (rank 4–8). Net params drop sharply; capacity loss is partially recovered. Free budget → more layers / bigger hidden.

**Expected gain.** 20–40% param reduction at <1% quality loss is achievable per ALBERT/LoRA literature. Reinvested: **0.005–0.010 BPB**.

**Rule-legality.** ✅ Fine.

**Complexity.** MEDIUM. Sharing is trivial (`self.shared_block = Block(...)`). The delta logic needs QAT-aware training; deltas themselves quantize differently than full weights. 1–2 days.

**Go/No-go.**
- **Go if:** Angle A (entropy coding) does NOT already yield ≥1.5MB savings. (If A works, we don't need F.)
- **No-go if:** A covers the capacity budget we need.

**Risk.** Medium. Sharing too aggressively hurts; deltas add code complexity.

**Priority: MEDIUM (fallback if A underdelivers).**

---

## 3. Ranked recommendation

| Rank | Angle | Expected BPB win | Effort | Risk | Decision |
|---|---|---|---|---|---|
| 1 | **A — Entropy coder** | 0.005–0.010 | 1 day | Low | **Build first** |
| 2 | **C — Data curation** | 0.003–0.010 | 0.5 day | Low (if legal) | **Build in parallel with A** |
| 3 | **B — Learned VQ/PQ** | 0.010–0.015 | 2–3 days | Med-High | **Prototype if A+C ship by Apr 23** |
| 4 | **F — Weight sharing+deltas** | 0.005–0.010 | 1–2 days | Medium | **Only if A underdelivers** |
| 5 | **D — SSM hybrid** | 0.002–0.008 | 2–3 days | High | **Skip unless ahead of schedule** |
| 6 | **E — Tokenizer** | 0.003–0.015 (risky) | 1–2 days | High (review) | **Skip for this competition** |

**Stacking order for final submission:**
`v1 baseline → + A (entropy coder) → + C (data curation) → + B or F (capacity reinvestment)`

If stars align, total gain = **0.018–0.035 BPB** over v1 (v1 ≈ 1.123 → target ≈ 1.088–1.105). That would be a clear record.

---

## 4. Work plan (Apr 14 → Apr 30)

### Planning phase (Apr 14–20) — in China, no code

- [ ] Apr 14 — this doc drafted; VPN set up before flying
- [ ] Apr 15 — **Rule deep-dive.** Verify data-ordering legality (Angle C). Verify tokenizer scrutiny criteria (Angle E). Confirm 10-min train/eval on 8×H800 equivalence.
- [ ] Apr 16 — **Angle A spec.** Write pseudocode for rANS coder, pick lib (`constriction` vs `torchac`), estimate bytes-saved empirically from v1 weight histograms (offline on phone: just need a histogram plot).
- [ ] Apr 17 — **Angle C spec.** Dedup pipeline design; curriculum signal source (pilot model vs zero-shot heuristic).
- [ ] Apr 18 — **Angle B spec (contingent).** PQ sub-vector size, codebook count per layer, QAT-VQ training recipe.
- [ ] Apr 19 — **Risk review.** Sanity-check each bet against the 10-min training cap. Fallback plan if A fails.
- [ ] Apr 20 — **Freeze plan.** Return home. Git branches ready: `v2-entropy-coder`, `v2-data-curation`.

### Execution phase (Apr 21–29) — local + H800

- Apr 21: Local prototype of Angle A (entropy coder) on v1 weights. Measure size.
- Apr 22: Angle C — dedup + curriculum preprocessing, local 1-shard smoke.
- Apr 23: Integrate A + C into `train_gpt.py`. Commit branch. 1-shard smoke on 5070 Ti.
- Apr 24–25: First H800 full run, 3 seeds. Measure BPB.
- Apr 26: Go/no-go on B vs F based on A+C results.
- Apr 27: Implement B or F.
- Apr 28: Second H800 full run, 3 seeds.
- Apr 29: Final seed average, submission prep.
- Apr 30: Submit.

---

## 5. Open questions to resolve this week

1. Does the eval harness fix the data shuffle seed? (Affects Angle C.)
2. Are 8×H800 and 8×H100 interchangeable for train-time rule compliance? (They're similar but H800 has reduced NVLink bandwidth — all-reduce timings differ.)
3. What's the actual per-row weight distribution in our v1 model? (Histogram dictates Angle A's gain.)
4. Is there a prior ANS/rANS submission on the leaderboard we haven't seen? (Check all records.)
5. How much do code bytes currently cost vs weight bytes in v1? (`len(train_gpt.py)` compressed, to know the true capacity budget.)

---

## 6. VPN setup for China — practical guide

**Background.** Mainland China's Great Firewall blocks `claude.ai`, Anthropic API endpoints, Google, GitHub (intermittent), Hugging Face, and most Western AI services. VPNs work, but the government actively detects and throttles common protocols. **Set everything up before you fly** — VPN provider websites themselves are often blocked from inside China.

### Recommended options (ranked by China reliability, Apr 2026)

| Option | Cost | China reliability | Setup difficulty | Notes |
|---|---|---|---|---|
| **Astrill VPN** | ~$30/mo | ⭐⭐⭐⭐⭐ (best) | Easy | The long-time favorite of expats in China. StealthVPN / OpenWeb protocols evade detection. |
| **ExpressVPN** | ~$13/mo | ⭐⭐⭐⭐ | Easy | Works most days; occasional outages during sensitive political windows. |
| **Self-hosted Shadowsocks / V2Ray (VLESS+Reality)** | ~$5/mo VPS | ⭐⭐⭐⭐⭐ | Medium | Most robust. Deploy on a VPS in Tokyo/Singapore/HK. Reality protocol is currently the hardest to block. |
| **Mullvad** | €5/mo | ⭐⭐ | Easy | Often blocked in China. Not recommended. |
| **NordVPN** | ~$12/mo | ⭐⭐⭐ | Easy | Inconsistent; "obfuscated servers" sometimes work. |

### My recommendation for your trip

**Primary: Astrill** (paid, easy, highest success rate for a ~1 week trip).
**Backup: Self-hosted V2Ray+Reality** on a $5/mo VPS in Singapore.
**Emergency backup: Let's VPN** free tier on mobile.

Carrying two independent VPN solutions is standard practice — if one gets blocked mid-trip (common during political events), you don't lose access.

### Pre-flight checklist (do before leaving your home country)

1. ✅ Subscribe + install Astrill on laptop, phone, tablet (all devices you'll use).
2. ✅ Download the app **from the Astrill website** (not App Store — Apple China removes VPN apps).
3. ✅ **Test it** by connecting to a Japan/Singapore server and loading `claude.ai` — confirm you can log in.
4. ✅ Rent a $5 VPS (Vultr or DigitalOcean, Singapore region) and install V2Ray+Reality. Write down the connection string.
5. ✅ Install a V2Ray client (`v2rayN` Windows, `v2rayNG` Android, `Shadowrocket` iOS — get the iOS app from a non-China App Store or before flying).
6. ✅ Download your **Claude Code auth token / API key** and store it offline — Anthropic's OAuth flow occasionally fails on VPN; having the token cached is insurance.
7. ✅ Download key repo docs (this `BRAINSTORM.md`, `README.md`, the rules) to your phone so you can read them offline if VPN drops.

### In-China operating tips

- Use a **Japan or Singapore** exit node; they have the lowest latency from China.
- If a server suddenly stops working, rotate to another one — don't assume the VPN itself is dead.
- Avoid connecting from hotel Wi-Fi when you can help it; cellular data (via roaming or a local SIM) often has better luck with VPN passthrough.
- Claude mobile app works identically to web once VPN is up.
- GitHub loads slowly even via VPN; batch your `git push`/`pull`.
- **Don't panic if `claude.ai` times out once** — try a different exit node first.

### What to do if *everything* is blocked

- SSH to a Singapore jump box (DigitalOcean, ~$5/mo) and run `claude` CLI there. SSH is rarely blocked.
- Bonus: that jump box is also useful as your V2Ray server, so it's dual-purpose.

### Cost summary

- Astrill: $30 for one month = fine for a trip.
- Singapore VPS: $5/mo.
- **Total: ~$35** for robust multi-redundant access.

---

## 7. Next actions (this session)

1. You review this doc on the flight / in China.
2. Flag any angle you want to cut or add.
3. Answer the 5 open questions in §5 when we have the info — I'll help dig through the repo to resolve them.
4. Once you approve the ranking, I start drafting the detailed spec for Angle A.

No code changes until Apr 21.
