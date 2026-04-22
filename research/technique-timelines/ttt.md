# TTT (Test-Time Training) — Technique Timeline

**Compiled:** 2026-04-21  
**Scope:** record track (`track_10min_16mb/`), merged SOTA → our baseline #1736  
**Our baseline:** PR #1736 (dexhunter, 1.06549 bpb)

---

## Background: what TTT is in this context

In parameter-golf, "TTT" means adapting model weights at **evaluation time** using the
validation corpus itself. The model sees each validation chunk, scores it (measures loss),
then updates weights via gradient descent before continuing to the next chunk.

This is legal under Issue #1017 provided the scoring step (which is used for the final
loss measurement) happens **before** the gradient update that would have used that
information — "score-first". The constraint is: you cannot update on a chunk and then
re-score the same chunk with the updated weights (that would be cheating).

---

## Phase 1 — Score-first TTT as canonical legal baseline (PR #1493)

**PR #1493** (bigbag, merged SOTA, 1.0810 bpb)

This is the baseline every subsequent TTT PR stacks on. Key TTT config:
- SGD optimizer, `lr=0.005`, `momentum=0.9`
- 3 passes over the validation corpus (3 "epochs")
- Cosine LR decay over those 3 epochs
- **Score-first**: each chunk is scored under `torch.no_grad()` BEFORE the gradient
  update, satisfying Issue #1017 Condition 3
- LoRA adapters trained (not full fine-tune) — keeps the 16MB artifact budget
- Sliding window: each SGD update is informed by the chunk just scored

The score-first mechanism is the key legal invention. Prior TTT approaches updated then
re-scored, which was ruled illegal. #1493 cleanly separates the two steps.

---

## Phase 2 — Doc-independent LoRA TTT (PR #1530)

**PR #1530** (samacqua, 1.07336 bpb)

**Core change:** Instead of a single shared LoRA adapter that slides through the corpus,
each validation *document* gets its own independent LoRA adapter trained from scratch.

Why this matters:
- Under the sliding-window approach, later documents in the corpus benefit from gradient
  signal accumulated across earlier documents. This creates implicit inter-document
  leakage — document N's scores are influenced by all documents 1…N−1.
- Doc-independent LoRA eliminates this: each adapter starts from zeros and only sees
  its own document's signal. No leakage → cleaner, more defensible TTT.
- Empirically: **+0.008 bpb improvement** over the sliding-window baseline.

Implementation: LoRA adapters are initialized fresh per document, run N gradient steps,
then discarded. The base model weights are not modified between documents. For efficiency,
samacqua batches LoRA forward passes where document lengths allow.

**Note:** This is the lineage that flows into our baseline. The sliding-window TTT from
#1493 is not in our #1736 baseline — doc-independent LoRA replaced it.

---

## Phase 3 — Phased global SGD (PR #1610)

**PR #1610** (romeerp, 1.07281 bpb, −0.00056 vs #1530)

**Core change:** After the per-document LoRA passes, pause and run a **global SGD
phase** over all already-scored documents, then resume scoring.

The insight: per-document LoRA adapters are good at local adaptation, but they can't
learn global patterns that span many documents. A separate global SGD pass over the full
scored corpus captures cross-document structure.

Config:
- `PHASED_TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000` — pause after 2000 documents for the global pass
- One global SGD phase (single pause point)

The global pass itself is score-first compliant: all documents in phase 1 were already
scored before the global update runs, so no document is re-scored with updated weights.

**−0.00056 vs #1530.** Modest gain — the global pass is capturing something real but
small. The single-phase design leaves room for more.

---

## Phase 4 — Multi-phase global SGD (PR #1626)

**PR #1626** (dexhunter, 1.07193 bpb, −0.00087 vs #1610)

**Core change:** Instead of one global phase, interleave **multiple global SGD passes**
throughout the corpus. The corpus is divided into N chunks; after each chunk is
doc-independently scored, a global SGD pass runs over all scored docs so far, then
scoring resumes.

Config from PR:
- 3 phases (later PRs try 4)
- `MATRIX_LR=0.026` — the learning rate for the global SGD passes (separate from the
  per-doc LoRA LR)
- Phase boundaries divide the corpus into equal thirds

Why better than single-phase: the global SGD pass at phase 1 updates weights before
documents in phase 2 are scored. So phase 2 documents are scored with a better-adapted
model. This is legal because phase 2 documents haven't been seen yet when the phase 1
global update runs.

This is the "PhasedTTT" or "multi-phase global SGD TTT" that is now in our baseline.

**Independent replication:** PR #1700 (jorge-asenjo, 1.07219) independently implemented
the same 3-phase multi-phase global SGD approach, arriving at essentially the same score
(within 0.0003 bpb). This double-confirmation makes the technique credible.

---

## Phase 4a — 4-phase scaling (PR #1727)

**PR #1727** (yahya010, 1.07217 bpb, 4 phases)

Tested 4 phases instead of 3. Result: **−0.00002 vs 3 phases**, well within noise
(SOTA std ~0.0002). Config: `PHASED_TTT_NUM_PHASES=4 QK_GAIN_INIT=5.25`.

**Key finding:** 3 → 4 phases is saturated. More phases beyond 3 are not free gains.

---

## Phase 5 — TTT / GPTQ incompatibility discovered (PR #1341 analysis)

Not a positive result — a blocker discovered by multiple competitors.

**Finding:** GPTQ quantization and TTT are incompatible at the mechanism level.

GPTQ works by performing column-wise Hessian-weighted error redistribution when
quantizing each weight column. The resulting quantized weights carry a compensatory
structure: the quantization error in column i is corrected for by adjusting subsequent
columns. This structure is **global and fragile** — it depends on the exact weight
values at quantization time.

When TTT's SGD pass modifies weights, it destroys this compensatory structure. The GPTQ
correction that made int8 weights behave like float16 weights is now invalid. The
effective quantization error jumps from corrected-small to uncorrected-large.

**Practical consequence for our work:** This is why specs 009, 010, 010b (SpinQuant
variants) showed no gain or regression. SpinQuant rotates the weight space before GPTQ
quantization; TTT then rotates the weights back (implicitly, via gradient updates),
negating the SpinQuant benefit. We marked SpinQuant as "TTT-absorbed."

More broadly: any technique that works by carefully arranging weight values at
quantization time (GPTQ, SpinQuant, SDClip, Hessian-weighted clipping) will be degraded
by TTT unless the technique is re-applied after each global SGD pass — which is
computationally infeasible in the 10-minute budget.

---

## Phase 6 — What's in our baseline #1736

Our baseline (PR #1736, dexhunter, 1.06549) combines:

1. **Doc-independent LoRA TTT** (from #1530 lineage) — per-document fresh adapters
2. **Multi-phase global SGD** (from #1626) — 3 interleaved global passes, `MATRIX_LR=0.026`
3. **Score-first** — fully legal, satisfies Issue #1017 Condition 3

The two components are not in conflict: doc-independent LoRA handles local adaptation
per document, and the global SGD phases capture cross-document structure. They operate
in complementary regimes.

Our baseline does **not** use:
- Pre-quant TTT (likely-illegal per Issue #1017 C3 — see #1735/#1738)
- More than 3 global phases (tested, shown saturated in #1727)
- Sliding-window TTT (superseded by doc-independent LoRA)

---

## Summary table

| Date | PR | Author | bpb | Key change | Status |
|---|---|---|---|---|---|
| 2026-01 | #1493 | bigbag | 1.0810 | Score-first TTT, SGD lr=0.005, 3 epoch cosine | **MERGED SOTA** |
| 2026-03 | #1530 | samacqua | 1.07336 | Doc-independent LoRA TTT, +0.008 vs sliding | closed/unmerged |
| 2026-03 | #1610 | romeerp | 1.07281 | Single phased global SGD (1 pause @ 2000 docs) | closed/unmerged |
| 2026-04 | #1626 | dexhunter | 1.07193 | Multi-phase global SGD, 3 phases, `MATRIX_LR=0.026` | closed/unmerged |
| 2026-04 | #1700 | jorge-asenjo | 1.07219 | Independent 3-phase multi-phase SGD replication | open |
| 2026-04 | #1727 | yahya010 | 1.07217 | 4 phases (vs 3) — saturated, within noise | open |
| 2026-04-19 | #1736 | dexhunter | 1.06549 | **Our baseline**: doc-indep LoRA + 3-phase global SGD + CaseOps + gates | open |
| 2026-04 | #1735 | unknown | 1.0429 | Pre-quant TTT — **likely-illegal** | disputed |
| 2026-04 | #1738 | unknown | 1.0354 | #1735 + CaseOps — **likely-illegal** | disputed |

---

## Key findings

1. **Doc-independent LoRA is the biggest single TTT gain** (~0.008 bpb over sliding-window).
   The cross-document leakage in sliding-window is real and hurts; eliminating it improves
   scores.

2. **Multi-phase global SGD adds ~0.002 bpb on top of doc-independent LoRA.** The gain
   is genuine (replicated independently by #1700) but smaller than the LoRA improvement.

3. **Phase count saturates at 3.** PR #1727 tested 4 phases and found −0.00002 vs 3
   phases. More phases are not worth the compute.

4. **TTT absorbs quantization-side levers.** GPTQ/SpinQuant compensatory weight
   structures are destroyed by TTT's SGD updates. Any quant-side PR should be assumed
   TTT-absorbed unless specifically shown otherwise.

5. **Pre-quant TTT is likely-illegal.** PRs #1735/#1738 run TTT before quantization,
   then re-quantize — this violates the score-first constraint and is under active
   dispute. The claimed 0.030+ bpb gains are implausibly large (>physics ceiling for
   legal TTT).

---

## Open questions (never cleanly ablated on our current stack)

1. **Per-doc LoRA rank**: What rank are the LoRA adapters in #1736? Is there room to
   increase rank (more capacity per document) or decrease it (faster, possibly more
   regularized)?

2. **MATRIX_LR sensitivity**: #1626 uses `MATRIX_LR=0.026`. This has not been swept on
   the full #1736 stack (which adds CaseOps + gates). The optimal LR may differ.

3. **Phase boundary placement**: Are equal thirds the optimal split? Could front-loading
   (more frequent global passes early in the corpus) help?

4. **Doc-independent LoRA LR**: The per-doc LoRA adapter LR (separate from `MATRIX_LR`)
   has not been cleanly ablated on the full #1736 stack.

5. **Score-first vs score-after on global passes**: The global SGD pass currently uses
   scores computed from the per-doc LoRA pass. Could re-scoring after the global update
   help? (Probably illegal under strict Issue #1017 C3 reading, but worth checking the
   exact constraint.)

6. **LoRA adapter targets**: Which weight matrices get LoRA adapters? Are there
   untested targets (e.g. applying LoRA to the gating matrices introduced in #1736)?

---

## Potential opportunities

### A. MATRIX_LR sweep on #1736 stack
**What:** Grid search `MATRIX_LR ∈ {0.018, 0.022, 0.026, 0.030, 0.034}` (current is
0.026, from the pre-gates/pre-CaseOps stack).  
**Why:** #1626 tuned this on a different model config. The attn-out gate and CaseOps
tokenizer change the loss landscape; optimal LR may have shifted.  
**Implementation:** One-liner env var change, no code. 5 smoke runs.  
**Estimated Δ:** 0.0002–0.0010 bpb (high uncertainty, could be zero).  
**Risk:** Low. Pure hyperparam.

### B. Per-doc LoRA rank ablation
**What:** Try `LORA_RANK ∈ {4, 8, 16, 32}` (assuming current is 8 or 16 — verify first).  
**Why:** Never ablated on our stack. Higher rank = more capacity but slower per-doc
passes; lower rank = more regularized, faster.  
**Implementation:** Likely a config change, may need code check to confirm the param name.  
**Estimated Δ:** Unknown. Could be positive or negative.  
**Risk:** Medium. Need to verify the param is actually configurable without code changes.

### C. Phase count 2 (regression check)
**What:** Try `PHASED_TTT_NUM_PHASES=2` — one fewer global pass than our current 3.  
**Why:** We know 3→4 is saturated, but we don't know the 2→3 delta on *our specific
stack*. If 2≈3, we save compute during TTT; if 2<3, confirms 3 is load-bearing.  
**Implementation:** One-liner. Cheap.  
**Estimated Δ:** Likely −0.0005 to 0 (regression). Worth knowing.  
**Risk:** Very low. Diagnostic value.

### D. LoRA adapter on gating weights
**What:** Check whether the attn-out gate and quant-gate weight matrices added in #1736
are included in the LoRA adapter target list. If not, add them.  
**Why:** #1736 introduced new learned gating matrices. Per-doc LoRA adaptation on these
matrices could capture document-specific gating behavior.  
**Implementation:** Requires code change to the LoRA target list. Moderate complexity.  
**Estimated Δ:** Unknown. Speculative.  
**Risk:** Medium. Needs code change, could interact with quant.
