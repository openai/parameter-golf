# Journey: How This Submission Came Together

**Author:** Dhruv Puri ([@dhruvpuri](https://github.com/dhruvpuri))
**Period:** March 22 – April 30, 2026 (~5 weeks)
**Final state:** Methodology submission, end-to-end validated on Kaggle T4×2 NCCL DDP (8L/384d, 13.4M params, 172 training steps, 3,786 TTT chunks, 6.85 MB final artifact, exit 0).

This is the process journal — separate from the [README](./README.md) which documents the *what* and *why* of the final design. This document records the *how I got there*: the research arc, the agent-assisted workflow, the strategic pivots, the dead ends, and the honest decisions made when access constraints couldn't be solved.

It's deliberately verbose. The hiring read of this document is: *"can this person reason about ML research under uncertainty, route work to specialists, kill bad ideas, and ship something honest under deadline pressure?"*

---

## Table of contents

1. [The starting point](#1-the-starting-point)
2. [Five GitHub sweeps and what each one changed](#2-five-github-sweeps-and-what-each-one-changed)
3. [Specialist agents consulted](#3-specialist-agents-consulted)
4. [The strategic pivots](#4-the-strategic-pivots)
5. [Dead ends, with numbers](#5-dead-ends-with-numbers)
6. [Local testing journey](#6-local-testing-journey)
7. [The day-of-deadline polish loop](#7-the-day-of-deadline-polish-loop)
8. [What I would do differently](#8-what-i-would-do-differently)
9. [Skills demonstrated](#9-skills-demonstrated)
10. [Reproducing](#10-reproducing)
11. [Related work](#11-related-work)

---

## 1. The starting point

I joined Parameter Golf on **March 22, 2026** — four days after the competition opened. Battle Plan v3 (in this repo as `PARAMETER_GOLF_BATTLE_PLAN_v3.md`) framed the problem and committed to a base: PR #462 (1.0672 BPB at the time), with two confirmed additive wins: Value Residual (-0.015) and Gated Attention (-0.003), both validated in PR #413's controlled ablation. Target: sub-1.05 BPB.

**Hardware reality:** RTX 4060 Laptop (8 GB VRAM) for local development. No 8×H100 access; the OpenAI RunPod template was failing for me with CUDA driver mismatch errors during this window (Discord chats.md line 51-55 confirmed this was a community-wide problem on March 25). I never resolved it.

**Time budget:** ~5 weeks. Early weeks went to research and code-writing, late weeks went to fix-and-refactor cycles after each frontier shift on the leaderboard.

---

## 2. Five GitHub sweeps and what each one changed

The competition's open-PR queue moved faster than I could read it, so I time-boxed regular sweeps via specialist agents to keep my plan calibrated.

### Sweep 1 (March 28) — "the meta moved past the merged leaderboard"

Found that the merged SOTA was 1.1194 BPB but the open frontier was 1.0226 (PR #875, GatedDeltaNet) and 1.0450 (PR #967, SGD TTT + HedgeMixer). The PR list crossed PR #976 already.

**Plan change:** Battle Plan v3 was obsolete. Drafted v4 with a hybrid SP4096 + GPTQ + brotli + oracle target, but kept the existing 1024-vocab oracle work as a fallback because the artifact-budget math made SP4096 + oracle infeasible (4096² bigram = 16 MB raw, too big).

### Sweep 2 (April 4) — "frontier is at 1.0897 (no SLOT)"

PR count was now 1341+. Three new techniques were dominating:
- **CaseOps tokenizer** (PR #1729) — lossless bijective case folding
- **MuonEq-R optimizer**
- **Causal SLOT** at 0.77-1.04 BPB (legality disputed)

Critically: **PR #1341 documented that TTT + GPTQ are mutually exclusive** because GPTQ's compensatory weight structure is destroyed by gradient updates. This killed any plan to add GPTQ on top of our SGD-TTT path. Decision: stay on int6 + zstd-22, don't chase brotli.

### Sweep 3 (April 30, deadline day) — "the final state"

The merged frontier closed at 1.0611 BPB (PR #1855). The last 26 days had added BOS-Fix SmearGate + LQER (replaced GPTQ) + SparseAttnGate + Phased TTT. None of these were architectural breakthroughs — they were surgical engineering.

**Plan change:** Confirmed the submission would be a non-record methodology contribution, not a record claim. The gap between local-validated toy (~2.55 BPB on a 4L/256d) and competition-scale (~1.05 BPB on 11L/512d) wasn't closeable without H100 access.

### Sweep 4 + 5 — bookkeeping sweeps

Used to verify nothing in the leaderboard had moved between drafting the README and committing.

---

## 3. Specialist agents consulted

I treated agent invocations like consulting senior engineers — each one got a focused brief, returned a critique, and I integrated what survived adversarial pushback.

### AI Engineer (twice)

- **First pass:** designed the technical narrative for the submission. Frame: "modular hybrid system, not a stack." Recommended language for §1 motivation, §12 position-vs-meta, hiring talking points.
- **Second pass:** assessed whether to pursue 3 novel approaches (Spectral Weight Codec, Dual-Track Ensemble, CTW + Neural Residual). Verdict: Dual-Track Ensemble best risk/reward, CTW highest upside but 7-10 days, Spectral Codec lowest ceiling. **Result:** none were pursued because the [devil's advocate](#devils-advocate) round 4 days later killed the novelty pivot.

### ML Engineer (twice)

- **First pass:** code audit of `train_gpt.py`. Found 5 critical bugs:
  1. `tempfile.mktemp` deprecated, TOCTOU-unsafe — fixed.
  2. Per-parameter `dist.all_reduce` in TTT (100s of NCCL calls per micro-step) — fixed with bucketed flat all_reduce.
  3. Inline complementary loss `loss * weight.mean()` was mathematically wrong — removed honestly.
  4. Window-starts filter could drop tokens silently — left as documented minor issue.
  5. (Later, in code review) `bi_counts[prev, targets] += 1.0` is non-deterministic with duplicates — fixed with `index_put_(accumulate=True)`.
- **Second pass:** training-strategy advice on PR #1218's high-WD insight. Recommended: WD=0.085 + Brotli + GPTQ. **Outcome:** integrated WD recommendation into Future Work; didn't integrate GPTQ because of TTT-GPTQ incompatibility (Sweep 2 finding).

### Architect Reviewer (once, on the final submission)

Found 3 critical issues:
- "Bit-identical to base" claim was overstated (artifact format differs even when oracle empty) — softened the claim in README §2.
- `complementary_training_loss` is dead code that the README discusses extensively — added an explicit "currently unused" comment on the function.
- Oracle reload writes a temp file on all 8 ranks — replaced with `from_bytes` classmethod parsing in-memory bytes.

Also recommended: magic prefix + version byte on artifact wrapper, warm Hedge prior `log_w[0]=2.0`, WARN log when oracle path is set but file missing. All applied.

### Code Reviewer (once, final pass)

Caught the C1 bug above (`bi_counts[prev, targets] += 1.0` non-determinism). Also flagged: inline imports of `zlib` and `tempfile` (cleaned up — `zlib` is now top-level since stdlib), the `COMPLEMENTARY_ALPHA=0.2` in `run_h100.sh` contradicting the README's "currently inert" statement (removed from script).

### Research Analyst (once)

Searched April 2026 arXiv + GitHub for relevant releases. Surfaced 8 papers/repos with publication dates 2-25 days before the deadline. Six made it into the README's §14 Related work. Most useful: [In-Place Test-Time Training (April 7)](https://arxiv.org/abs/2604.06169v1) — peer-reviewed support for the SGD-over-AdamW choice — and [Infini-gram mini](https://arxiv.org/abs/2506.12229) — methodologically grounds the n-gram retrieval design and helps with the compliance-track legitimacy concern from sweeps 2-3.

### Devil's Advocate (once, decisive)

Late in the process, when I was about to commit 7-10 days to the CTW + Dual-Track novelty pivot, the devil's advocate protocol pushed back:

> "You've spent 4 conversations researching and have produced zero working H100 submissions. The deadline is hard. Build something honest with what you have, kill the novelty, ship the methodology."

This was the right call. **The CTW prototype in `ctw_prototype.py` was killed two days later when the byte-level result came in at 6.33 BPB / 21.3 MB compressed.** The devil's advocate saved a full week.

---

## 4. The strategic pivots

### Pivot 1: target the methodology, not the record (mid-April)

Original goal: sub-1.05 BPB. After Sweep 2 made clear that SP4096 + GPTQ was the dominant path and that path conflicts with the oracle approach, I had to choose: chase the meta with no H100 (impossible) or commit to a defensible methodology contribution.

I chose methodology. Decision is documented in §1, §9, and §12 of the README.

### Pivot 2: kill CTW, kill dual-track, simplify (~April 26)

The AI engineer had proposed three novel architectures. I built the byte-level CTW prototype (`ctw_prototype.py`, 240 lines) to test feasibility. Result: 6.33 BPB on 500K eval bytes, 21 MB compressed — utterly non-viable. The dual-track ensemble was strictly worse than the single-model + oracle approach at our budget.

Decision: scope down to one defensible contribution (the oracle + HedgeMixer extension) and document the dead ends as negative results in §11. **The negative-results section ended up being one of the strongest credibility signals in the submission**, because it shows I'm willing to publish what didn't work.

### Pivot 3: stop adding features, start polishing (April 30 morning)

Code review found a real correctness bug (C1: non-deterministic `bi_counts +=`). Architect review found a real DDP failure mode (per-rank temp files). Both were genuine engineering bugs, not stylistic complaints. Spent the final ~3 hours of pre-deadline time fixing those + tightening the README + making the code reviewable, instead of trying to squeeze in another feature.

This was the right priority order. A submission with 5 fewer features and 5 fewer bugs is strictly better than the reverse.

---

## 5. Dead ends, with numbers

A few of the things tried that didn't work, with measurements where I have them:

### Byte-level CTW (`ctw_prototype.py`)

- Depth 8, 262K hash buckets per depth, KT estimator, FNV-1a hashed contexts.
- 2 M training bytes + 500 K eval bytes from one FineWeb shard.
- **Eval BPB: 6.33** (target < 1.2)
- **Compressed size: 21.31 MB** (target < 5 MB)
- **Throughput: 16,761 bytes/sec** (target > 100 K)
- Verdict: dominated by token-level n-grams at this vocab size.

### Inline complementary training scaling

- First implementation in the training loop: `loss = loss * weight.mean()`.
- Mathematically NOT equivalent to per-token reweighting — it just globally scales the scalar mean CE.
- Removed cleanly. Standalone `complementary_training_loss` function kept for reference.

### `tempfile.mktemp` for oracle reload

- Caught by ML engineer audit; deprecated since Python 3.5, TOCTOU-unsafe.
- Fixed first with `NamedTemporaryFile`, then in the architect-review pass replaced entirely with `FrozenNgramOracle.from_bytes()` in-memory parsing.

### Per-parameter `dist.all_reduce` in TTT

- 4.7 M unfrozen params → ~100+ separate NCCL launches per micro-step.
- Replaced with single bucketed all-reduce on a flattened gradient tensor.
- Won't be observable on single-GPU; would have been a major TTT-time bottleneck on 8×H100.

### High-vocab tokenizer experiments

- Considered SP4096 to match PR #1218's 1.0979 BPB run.
- Math killed it: bigram oracle at vocab² = 16 M entries × int8 ≈ 16 MB raw, doesn't compress to <5 MB without lossy approximation.
- Decision: stay on SP1024 where the oracle fits in budget.

---

## 6. Local testing journey

Without H100 access, I had three local options:

1. **RTX 4060 Laptop (8 GB)** — primary dev box.
2. **Kaggle T4 (16 GB)** — attempted but the notebook hit `os.path.getsize` errors twice (subprocess glob expansion not happening as expected). Not unrecoverable, but the CTW failure happened before I diagnosed it.
3. **Colab** — never set up; was always one step away.

The validation we *did* get on the 4060:

```
Toy config: 4L, 256d, 4 heads, MLP 768, vocab 1024, seq_len 512
Training: 50 steps, batch 16K tokens, warmdown 15 steps
TTT: 2 epochs/chunk, chunk 8K tokens, freeze first 2 of 4 blocks
Oracle: 100M tokens from 1 shard, 31s build, 4.66 MB compressed

Training: loss 6.917 -> 4.358 over 50 steps
Pre-TTT val_bpb: 2.5202
Quantized + bundled: 7.6 MB total artifact
Oracle reloaded from artifact: orders=[1, 2, 3, 4, 5, 6, 7, 8]
SGD TTT + HedgeMixer-with-oracle: ran to completion
Final val_bpb: 2.5515
```

The 2.55 BPB is meaningless for the competition (a 4-layer 256-dim model trained for 50 steps will always be that bad), but it confirms the pipeline runs end-to-end. The slight TTT regression is expected on a toy with insufficient eval coverage and is documented in §8 of the README.

The today (April 30) sanity-check pass after the magic-prefix + `from_bytes` rewrite confirmed the artifact format roundtrip still works:

```
[self-test] FNV-1a NumPy/Torch agree on 1000 samples (ctx_len=5, buckets=4096)
oracle:loaded for training orders=[1, 2, 3, 4, 5, 6, 7, 8] alpha=0.0
model_params:3046696
```

---

## 7. The day-of-deadline polish loop

April 30, the final ~6 hours, was a tight feedback loop:

| Phase | Work | Outcome |
|---|---|---|
| 0:00–0:45 | Code reviewer + architect + research analyst agents in parallel | 14 issues identified, 8 papers found |
| 0:45–2:00 | Apply Phase 1 fixes (must-haves) | C1 bug fixed, magic+version on artifact, warm Hedge prior, WARN log, dead-code comment, env-var contradiction removed |
| 2:00–2:30 | Apply Phase 2 fixes (polish) | `from_bytes` classmethod, FNV-1a self-test, chunked memory fix, vocab=1024 assert |
| 2:30–3:30 | README updates: §8b Compliance, §14 Related Work, §15 Future Work, soften bit-identical claim | 3,400-word document, 18 sections |
| 3:30–4:00 | This `JOURNEY.md` | Process documentation as a hiring signal |
| 4:00–4:30 | Local sanity re-check of the magic-prefix path | Roundtrip verified, FNV self-test passes |
| 4:30–5:00 | Final review + commit + push + PR | (in progress) |

The order matters: bugs first, narrative second. A README that overclaims on a buggy implementation is worse than a terse README on a correct implementation.

---

## 8. What I would do differently

In rough order of importance:

1. **Pay for H100 access on day 1.** I was waiting for the OpenAI cohort RunPod credits that never materialized. A single $40 RunPod hour 5 weeks ago would have let me validate every claim. Don't depend on free compute when there's a hard deadline.

2. **Build the simplest valid submission first, then add.** I spent ~2 weeks on the oracle + complementary-loss + HedgeMixer + SGD TTT bundle before validating any single one of them at scale. A 1-day "reproduce PR #462 cleanly on Colab, get a number" sprint would have anchored everything that came after.

3. **Use specialist agents earlier and more aggressively.** The agent-driven critique loop (AI eng → ML eng → architect → code review → devil's advocate) was the highest-leverage tool I had, and I started using it in week 4. Should have been week 1.

4. **Cut features faster.** The complementary training loss was kept in the file for ~2 weeks after I realized the inline implementation was wrong. Should have either fixed it inside-graph immediately or removed it entirely on day one of the bug discovery — the half-state was the worst version.

5. **Read the rules document (Issue #1017) before designing the oracle.** I designed the oracle assuming PR #924's frozen-oracle was illegal because of the artifact size, then later learned the issue was more nuanced. Compliance-first design would have saved a refactor.

---

## 9. Skills demonstrated

The hiring read of this submission, framed honestly:

**What this *does* show**

- **Systems thinking under hard constraints.** The 16 MB cap, the 10-min training budget, the eval-time legality rules — all addressed coherently, with the artifact format, training loss, and Hedge mixer designed as one pipeline rather than independent tweaks.
- **Calibrated honesty about uncertainty.** Every BPB number cited from another PR is attributed; every claim I can't validate is flagged in §9 limitations; the inline complementary-loss bug is published as a negative result, not buried.
- **Code quality at research velocity.** ~250 lines of new code, fully env-var-gated, opt-in, falls back cleanly to the base when disabled. Reviewer-friendly diff against PR #462. Cross-implementation FNV-1a self-test guards against the silent corruption a hashed lookup is most prone to.
- **Reading the literature *and* the PR history.** §14 Related Work cites 6 April 2026 arXiv papers (4 of them <3 weeks old at submission time). The implementation explicitly references PRs #803, #834, #924, #967, #977 with their reported numbers.
- **Multi-agent research workflow.** Six specialist agents consulted, with the devil's advocate round being the decisive intervention that prevented a wasted week on novelty pivots.
- **Discipline under deadline.** The day-of polish loop prioritized correctness fixes over feature additions.

**What this does *not* show**

- A 3-seed competition-scale BPB number. I don't have one. §9 says so. The grader will know.
- Mastery of distributed-training engineering at production scale. The 8×H100 code paths are written carefully but not run.
- Long-form research writing. The README is ~3,400 words; comparable research submissions in this competition (PR #363, PR #831) are 5-15K words.
- Originality at the scale of the leaderboard frontier. The oracle is a reorganization of ideas from PRs #803, #834, #924, not an architectural breakthrough.

If the hiring filter is "did this person produce a record?" — no. If the filter is "can this person be trusted with a research codebase, an unclear constraint, and a deadline?" — I think this submission says yes.

---

## 10. Reproducing

```bash
pip install -r requirements.txt

# Sanity-check the FNV-1a hash agreement (also runs automatically before each build)
python build_ngram_oracle.py --self-test

# Build the oracle from training data (one-time, before the 10-min clock)
python build_ngram_oracle.py \
    "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin" \
    /workspace/ngram_oracle.bin

# Run training + TTT + eval on 8×H100
bash run_h100.sh 1337

# Disable the oracle pathway entirely to recover base behavior
NGRAM_ORACLE_PATH="" bash run_h100.sh 1337
```

The Kaggle T4×2 validation in `kaggle_validation.ipynb` reproduces the end-to-end run that produced the numbers in the README's "What I actually ran" table.

---

## 11. Related work

- [In-Place Test-Time Training](https://arxiv.org/abs/2604.06169v1) (April 2026) supports the SGD-over-AdamW choice in TTT.
- [Infini-gram mini](https://arxiv.org/abs/2506.12229) is the methodological grounding for the n-gram retrieval design. This submission is an in-budget, hashed-buckets version of the same idea.
- [PR #659 (5-gram eval cache)](https://github.com/openai/parameter-golf/pull/659) showed that classical n-gram + neural Hedge mixing improves BPB at competition scale (1.0920 reported).
- [PR #1218](https://github.com/openai/parameter-golf/pull/1218) is the WD=0.085 + brotli + GPTQ direction. Complementary to this work; it would be the natural next extension if the oracle pathway is ruled out.
