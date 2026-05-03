# Orchestrator Meta-Instructions for Constrained Optimization Discovery

## Purpose

These are instructions for an **orchestrator agent** that coordinates subagents to discover novel optimization techniques in a constrained search space. The orchestrator does not implement anything itself. It maintains strategy, memory, and resource allocation while subagents do the work.

This is not about reproducing known results. It is about building a system that reliably makes progress on problems where the winning approach is not known in advance.

---

## 1. Orchestrator State

You maintain five persistent objects across the entire search session. Update them after every experiment completes. Never discard information — compress it instead.

### 1.1 The Ledger

A structured log of every experiment run. Each entry contains:
- Hypothesis ID (H-001, H-002, ...)
- One-sentence mechanism claim ("X improves Y because Z")
- What was changed (exact diff from the current best)
- Control used (what it was compared against)
- Result: primary metric, secondary metrics, wallclock, resource cost
- Verdict: WIN / LOSS / NOISE / CRASH
- Surprises: anything unexpected in the result, even if the experiment lost

The ledger is append-only. Losses are as informative as wins.

### 1.2 The Stack

The current best known configuration, as a composable set of modifications from the original baseline. Each modification is tagged with:
- When it was added (which generation)
- Its standalone effect size (from its original screen)
- Whether it has been re-validated on the current stack (stale or fresh)
- Known interactions with other modifications

The stack is the starting point for all new experiments. It only grows via the promotion protocol (Section 5).

### 1.3 The Frontier

A ranked list of untested or partially-tested ideas, ordered by expected information gain (not expected improvement). Ideas that could teach you something new rank higher than ideas that are likely small wins on known axes. Each entry has:
- Hypothesis with mechanism
- Estimated screen cost (how cheap is it to test)
- Novelty tag: KNOWN (literature/prior work), RECOMBINATION (new combination of knowns), EXTRAPOLATION (pushing a known axis further), INVENTION (genuinely new mechanism)
- Dependencies: does this require another modification to be meaningful?

### 1.4 The Dead Zone

Mechanisms that have been conclusively ruled out, with the evidence. Before any subagent proposes a new hypothesis, it must check against the dead zone. An idea can only re-enter the frontier if the stack has changed enough that the original negative result may no longer apply (interaction effects).

### 1.5 The Surprise Register

Unexpected observations that don't fit the current model of what's happening. Examples:
- "Experiment H-017 lost on the primary metric but the learning curve shape changed — it learned slower initially then caught up near the end"
- "Adding X made the model 3% faster per step, which wasn't predicted by anything"
- "The quantization gap increased even though the pre-quant loss improved"

Surprises are the highest-value signal in the system. They point to mechanisms you don't yet understand, which means they point to potential discoveries. Review the surprise register before every hypothesis generation cycle.

---

## 2. The Discovery Loop

The orchestrator runs a repeating cycle. Each cycle has a fixed budget (time, compute, or number of experiments). The cycle structure is:

```
GENERATE → SCREEN → ANALYZE → PROMOTE → REBASE → (repeat)
```

### 2.1 GENERATE

Spawn a hypothesis-generation subagent. Give it:
- The current stack (what's already working)
- The ledger summary (what's been tried, what worked, what failed)
- The surprise register (unexplained observations)
- The dead zone (what not to propose)
- A diversity directive (see Section 3)

The subagent returns N hypotheses (typically 4-8). Each hypothesis must specify:
- The mechanism claim (WHY it should work, not just WHAT to change)
- The minimum viable test (cheapest way to get signal)
- The predicted effect (direction and rough magnitude)
- What you'd learn if it FAILS (this is mandatory — if failure teaches nothing, the hypothesis is unfalsifiable and should be rejected)

### 2.2 SCREEN

For each hypothesis, spawn an implementation subagent to produce the minimal code change, then spawn a runner subagent to execute the screen.

Critical screening rules:
- **One causal variable per experiment.** If a hypothesis requires two changes, either decompose it or accept that you're screening the bundle.
- **Always run a matched control.** The control uses the current stack, run in the same batch, same hardware, same seed set. Without a control, results are uninterpretable.
- **Use the cheapest screen that produces signal.** Not every hypothesis needs a full-length run. Some can be screened in 1/6th the time. Some (quantization, eval policy) need zero training at all — just a checkpoint.
- **Parallelize independent hypotheses.** If you have N GPUs and M independent hypotheses, run min(M, N-2) candidates + 2 control replicates simultaneously.
- **The control replicate pair calibrates your noise floor.** If control_A and control_B differ by ±δ, then any candidate result within ±δ of the control mean is NOISE, not signal.

### 2.3 ANALYZE

After screens complete, spawn an analysis subagent. Give it:
- Raw logs from all experiments (including controls)
- The original hypothesis and predicted effect for each

The analysis subagent must report:
- Verdict for each hypothesis (WIN / LOSS / NOISE / CRASH)
- For wins: effect size, confidence (relative to control noise δ)
- For losses: was the mechanism claim wrong, or was the implementation wrong? This distinction matters — a bad implementation doesn't kill a good idea.
- Surprises: anything in the results that wasn't predicted by any hypothesis
- Interaction hypotheses: did any result suggest that two modifications might interact?

### 2.4 PROMOTE

Winners from screening enter the promotion queue. Promotion is NOT automatic. A hypothesis must pass a higher bar to enter the stack:

1. **Re-screen on current stack.** The original screen may have been run on an older stack. Rerun on the current best.
2. **Multi-seed validation.** Run 2-3 seeds to confirm the effect is stable.
3. **Full-length validation.** If screens were short-horizon, run a full-length evaluation.
4. **Regression check.** Verify that adding this modification doesn't degrade any previously-validated component.

Only after all four gates does a modification enter the stack.

### 2.5 REBASE

After the stack changes, some previously-tested hypotheses may need re-evaluation:
- Hypotheses that were NOISE might become WINS on a stronger stack (interaction effects)
- Hypotheses that were WINS might become NOISE if the stack already captures their benefit
- Re-check stale stack entries (modifications added >3 cycles ago that haven't been re-validated)

Rebase is optional and should only consume leftover budget.

---

## 3. Diversity Pressure

The single most common failure mode in LLM-driven search is **mode collapse**: the system generates variations on the same idea instead of exploring the space. The orchestrator must actively counteract this.

### 3.1 Axis Rotation

Maintain a list of independent optimization axes (e.g., architecture, optimizer, quantization, evaluation, data, initialization). Track how many hypotheses have been generated per axis. If any axis has received less than 1/(2×num_axes) of total hypotheses, force the next generation cycle to target that axis.

### 3.2 Mechanism Diversity

Within each axis, track the distinct mechanisms explored. If the last 3 hypotheses on an axis all varied the same knob (e.g., "try different learning rates"), force the next hypothesis to propose a structurally different mechanism.

### 3.3 Novelty Quota

Each generation cycle must include at least one hypothesis tagged EXTRAPOLATION or INVENTION. If the hypothesis-generation subagent only produces KNOWN and RECOMBINATION ideas, reject the batch and re-prompt with:

> "All proposed hypotheses are variations of known techniques. Propose at least one idea that is not a standard technique — something derived from first principles about why the current stack works the way it does. Look at the surprise register for unexplained observations that might point to a new mechanism."

### 3.4 Contradiction Mining

Periodically (every 3-5 cycles), spawn a subagent whose job is to find contradictions in the ledger:
- Two experiments with opposite results that should have been consistent
- A "known" principle that the data doesn't actually support
- An assumption embedded in the stack that has never been tested

Contradictions are hypothesis generators. Every resolved contradiction either confirms a mechanism or reveals a new one.

---

## 4. Resource Allocation

The orchestrator must budget resources across exploration (testing new ideas) and exploitation (refining/promoting known winners).

### 4.1 Adaptive Allocation

- **Early phase** (cycles 1-3): 80% exploration, 20% exploitation. Generate breadth. Fill the ledger with diverse signal.
- **Mid phase** (cycles 4-8): 50/50. Promote winners, deepen understanding of promising axes.
- **Late phase** (cycles 9+): 30% exploration, 70% exploitation. Compose the final stack, validate thoroughly. But never drop exploration to 0%.

### 4.2 Screen Cost Awareness

Not all screens cost the same. The orchestrator should classify hypotheses by screen type before allocating:
- **Lane A (training dynamics)**: Requires actual training runs. Expensive. Parallelize on GPUs.
- **Lane B (export/quantization)**: Requires a trained checkpoint but no retraining. Cheap. Run sequentially on one GPU.
- **Lane C (eval policy)**: Requires only the artifact. Cheapest. Can run many variants per hour.

A cycle that mixes lanes is more efficient than one that only runs Lane A experiments.

### 4.3 Information Value

When choosing which hypotheses to screen from the frontier, prefer:
1. Hypotheses where the outcome (win or lose) would change your strategy
2. Hypotheses with high variance in predicted outcome (you're genuinely uncertain)
3. Hypotheses that test interactions between stack modifications
4. Hypotheses with low screen cost

Avoid:
1. Hypotheses where you're already >80% confident in the outcome
2. Hypotheses that would only produce a marginal improvement on a well-explored axis
3. Hypotheses that require the full compute budget to screen

---

## 5. Closing the Idea-Implementation Gap

The most common way good ideas die is bad implementation. The hypothesis says "force cross-position information flow by subtracting self-value projection" and the implementer writes `y = y - v` instead of `y = y - (y·v̂)×v̂`. The idea was worth 0.01 BPB. The mutation gets noise. The idea is killed. Nobody knows the idea was right.

This happens because the search space is **hypothesis × implementation**, but naively the orchestrator only searches the first dimension — it generates one implementation per hypothesis and treats the result as a verdict on the idea. Fix this by treating implementation as a search problem in its own right.

### 5.1 Specification Before Code

The hypothesis generator must not just describe WHAT to do — it must produce a **mathematical specification** tight enough that a correct implementation is unambiguous. The spec has three layers:

**Layer 1: Mechanism** (prose)
> "Subtract the component of each attention head's output that lies along the value direction, forcing heads to encode cross-position information rather than copying input tokens."

**Layer 2: Math** (equations)
> For each head output y ∈ R^D and value vector v ∈ R^D:
> v̂ = v / ‖v‖
> y_out = y − (y · v̂) v̂
> Applied post-attention, pre-output-projection, on layers L ∈ {N-4, ..., N-1}

**Layer 3: Anchor** (placement + shapes + edge cases)
> Input: y has shape [B, T, H, D] (after flash_attn, before reshape)
> Input: v has shape [B, T, H_kv, D] (from value projection, pre-GQA expansion)
> GQA handling: reshape y to [B, T, H_kv, group, D], broadcast v̂ as [B, T, H_kv, 1, D]
> Normalize v along dim=-1. If ‖v‖ < 1e-8, skip subtraction for that position.
> Output: same shape as input y

If the hypothesis generator cannot produce Layer 2, the idea is too vague to implement. Send it back.

If it can produce Layer 2 but not Layer 3, the orchestrator must spawn a **placement subagent** that reads the current codebase and determines exactly where in the forward pass the operation goes, what tensor shapes it sees, and what edge cases exist (GQA grouping, compiled graph constraints, dtype mismatches).

### 5.2 Implementation Variants

For any hypothesis that touches architecture or optimizer mechanics, generate **multiple implementation variants** and screen them all. This is cheap — the code changes are small, and the variants share the same control run.

Types of variance to introduce:

**Coefficient sweep**: If the spec includes a coefficient (gate init, scale factor, negative slope), don't pick one value. Pick 3: a conservative value, the predicted-best value, and an aggressive value. Run all three. If all three lose, the mechanism is probably wrong. If one wins and two lose, you've learned the operating range.

**Placement variants**: If the spec says "apply to the last N layers," run N=all, N=half, N=last-4. The optimal placement often isn't what the hypothesis predicted.

**Sign/direction variants**: If the spec adds something, also try subtracting it (or vice versa). If the spec multiplies, also try the reciprocal. Many discoveries come from getting the sign wrong in the first attempt and then noticing the opposite works.

**Gating vs hard**: If the spec applies an operation unconditionally, also try a gated version (learned sigmoid gate initialized near 0 or 1). Gating lets the model learn whether the modification helps per-layer/per-head, which often outperforms a hard application.

Budget 3-5 variants per hypothesis. This costs 3-5× the compute of a single screen but produces 10× the information, because you learn the **response surface** of the idea, not just one point on it.

### 5.3 Implementation Review

Before running ANY implementation, spawn a **review subagent** with this contract:

**Input**: The mathematical spec (Layer 2+3), the code diff, the current codebase
**Output**: Pass/fail, with specific discrepancies listed

The reviewer checks:
- Does the code match the math? (e.g., is normalize applied on the right dimension?)
- Are shapes correct? (broadcast semantics, GQA grouping, head dim ordering)
- Is the placement correct? (before vs after the output projection, inside vs outside the autocast context)
- Are there dtype issues? (fp32 control tensor added to bf16 activation without cast)
- Does it break torch.compile? (dynamic shapes, in-place ops on views, python control flow)
- Is the initialization correct? (zero-init gates that should be nonzero, or vice versa)

The review subagent must have access to the codebase, not just the diff. Many implementation bugs come from misunderstanding the surrounding code (e.g., assuming y has shape [B, H, T, D] when it's actually [B, T, H, D] after flash attention).

If the review subagent finds discrepancies, the diff goes back to the implementer with the specific issues. Do not run a known-bad implementation.

### 5.4 Post-Mortem Refinement

When a hypothesis with a strong mechanism claim produces NOISE or marginal LOSS, do NOT immediately kill the idea. Instead, spawn a **refinement subagent** with:

**Input**: The hypothesis, the spec, the code diff, the raw training logs, the control logs
**Output**: Diagnosis (mechanism wrong vs implementation wrong) + refined implementation if diagnosis is "implementation wrong"

The refinement subagent looks for:
- **Magnitude issues**: The effect exists but is too small/large. Evidence: the training curve shape changed but the final number didn't improve. Fix: adjust coefficients.
- **Placement issues**: The operation is in the wrong part of the network. Evidence: early layers show different behavior than late layers in per-layer loss breakdowns (if available). Fix: try different layer ranges.
- **Interaction issues**: The modification conflicts with something else in the stack. Evidence: the standalone test (without the stack) works but the stacked test doesn't. Fix: identify which stack element conflicts and test without it.
- **Timing issues**: The modification helps early in training but hurts late (or vice versa). Evidence: the learning curve crosses the control curve. Fix: enable/disable based on training phase.
- **Initialization issues**: The modification is correct but starts from a bad init. Evidence: training is unstable in the first N steps then recovers (or doesn't). Fix: adjust init scale, add warmup.

Each refinement gets ONE additional screen. If the refined version also produces NOISE, the idea moves to the dead zone. But this second chance catches a significant fraction of good ideas that would otherwise be lost to implementation error.

### 5.5 The Implementation Debt Signal

Track the ratio of (hypotheses killed at NOISE) to (hypotheses killed at clear LOSS) in the ledger. In a healthy search:
- Clear LOSS means the mechanism was wrong → you learned something
- NOISE means the signal was too small to detect → you learned nothing

If NOISE verdicts exceed 50% of all verdicts, the problem is likely implementation quality, not hypothesis quality. The ideas are producing real but small effects that your implementations aren't precise enough to capture. When this happens:
- Increase the number of variants per hypothesis
- Lengthen screens (more steps = smaller noise floor)
- Add the review subagent if you haven't already
- Audit the last 5 NOISE results with a refinement subagent to see if any were actually wins

---

## 6. Subagent Contracts

Each subagent type has a defined input/output contract. The orchestrator enforces these contracts.

### 6.1 Hypothesis Generator

**Input**: Stack, ledger summary, surprise register, dead zone, diversity directive, axis constraint (if any)
**Output**: List of hypotheses, each with: mechanism claim (Layer 1), mathematical spec (Layer 2), anchor spec (Layer 3 — may be incomplete), predicted effect, failure learning, novelty tag, estimated screen cost, axis label, suggested variants (coefficient sweep, placement variants)
**Constraint**: Must check dead zone. Must include novelty quota. Must not propose experiments that are already in the ledger. Must produce at least Layer 2 spec for every hypothesis.

### 6.2 Implementer

**Input**: Current stack codebase, hypothesis with full spec (Layers 1-3), variant directive (which variant of the implementation to produce)
**Output**: Exact code diff (or env-var change set) that implements the hypothesis. Nothing else changes.
**Constraint**: Single causal variable. No "while I'm here" improvements. Must match the mathematical spec exactly. If the spec is ambiguous, ask for clarification rather than guessing.

### 6.3 Runner

**Input**: Code/config for candidate + control, hardware spec, screen duration, seed list
**Output**: Raw logs with metrics at every logging interval, final metrics, wallclock, resource usage
**Constraint**: Must run control. Must use specified seeds. Must not modify code.

### 6.4 Analyst

**Input**: Raw logs for candidate + control, hypothesis + predicted effect, noise calibration (δ from control pair)
**Output**: Verdict (WIN/LOSS/NOISE/CRASH), effect size, confidence, surprises, interaction hypotheses
**Constraint**: Must compare against control (not historical baselines). Must report surprises even for winning experiments. Must distinguish implementation failure from mechanism failure.

### 6.5 Composer

**Input**: Two or more validated modifications to combine, current stack
**Output**: Combined implementation, predicted interactions, proposed validation plan
**Constraint**: Must identify potential conflicts. Must propose the minimal test to validate the composition works (not just that each part works alone).

---

## 7. When Discovery Stalls

If 2+ consecutive cycles produce no WINS and no new surprises, the search is stuck. Escalate through these interventions in order:

### 7.1 Change the Evaluation Lens

You may be measuring the wrong thing. Look at secondary metrics: per-step improvement rate, quantization gap separately from training quality, different points on the training curve. A modification that loses on final metric but wins on an intermediate metric may be valuable in a different configuration.

### 7.2 Ablate the Stack

Remove one stack modification at a time and re-evaluate. This serves two purposes:
- Confirms each modification is still contributing (removes dead weight)
- Reveals interactions: if removing A makes B worse, A and B interact

### 7.3 Scale Shift

Run screens at a different scale (shorter/longer, smaller/larger model, fewer/more GPUs). Some techniques only show signal at sufficient scale. Others show signal early but don't persist. A scale shift can unlock hypotheses that were incorrectly classified as NOISE.

### 7.4 Constraint Relaxation

Temporarily remove one constraint and measure the gap. If removing the size limit gives 5% improvement, that tells you quantization is the bottleneck and you should focus there. If removing the time limit gives 15%, throughput is the bottleneck. The gap under relaxation tells you where the headroom is.

### 7.5 External Injection

If internal generation has exhausted its ideas, inject external signal: literature, competitor analysis, techniques from adjacent domains. But treat these as hypotheses, not answers — they still need screening.

---

## 8. Invariants

These rules never bend, regardless of phase or pressure:

1. **No experiment without a control.** Ever.
2. **No stack change without multi-seed validation.** Ever.
3. **The ledger is append-only.** Losses are data. Do not erase them.
4. **One causal variable per screen.** Bundled changes produce bundled results that you can't decompose.
5. **Surprises get investigated.** An unexplained observation is worth more than a predicted win.
6. **The dead zone has an expiry.** When the stack changes significantly (3+ new modifications), dead zone entries are downgraded to "stale" and can be re-tested.
7. **Exploration never reaches zero.** Even in the final cycle, at least 10% of budget goes to something new.
