# SSM Worktree Setup Plan — instructions for the executing agent

**You are the executing agent.** A previous design pass wrote a heavier version of this plan; a critical reviewer pruned it; the user reviewed both. This is the slimmed final version. Execute Phase A onwards in §4. Update the checkboxes as you complete each item.

**Audience contract**: every instruction was written assuming you read this cold. If anything is unclear, the design pass failed you — ask the human (Tony) before guessing. Don't improvise; the sketches in §3 are deliberate, not drafts to improve.

**Last updated**: 2026-04-26 (final after design + review + second-review + third-review)
**Branch**: autoresearch-ssm
**Worktree**: /Users/tonyliu/Desktop/projects/parameter-golf-ssm

---

## Notes for the executor (read once before Phase A)

- **Read these files first to orient** (cold-read prerequisite):
  - `program.md` (current; you'll edit sectionally per §3.1)
  - `journal.md` (current; you'll reset per §3.2)
  - `train_gpt.py` lines 290-340 (the `CONTROL_TENSOR_NAME_PATTERNS` region — referenced in §3.1(c))
  - `.claude/skills/promote/SKILL.md` and `.claude/skills/search_journal/SKILL.md` (you'll light-edit per §3.6 and §3.7)
- **Line numbers in this plan are working-tree-relative**, not HEAD. `program.md` and `run_experiment.sh` have uncommitted improvements (per `git status`: `M program.md`, `M run_experiment.sh`). The line numbers I cite (e.g., `train_gpt.py` line 304, `program.md` line 165) are accurate against the current working tree. If a line-number reference doesn't match what you Read, find the section by content match rather than assuming the plan is wrong — the worktree may have drifted slightly.
- **Existing uncommitted changes in `program.md` and `run_experiment.sh` are NOT yours to revert.** They're pre-SSM-setup improvements (skill references in program.md; plan.md template-gate in run_experiment.sh). Preserve them. Phase F's "do NOT commit unless asked" means the user will decide whether to bundle these with the SSM-setup commit or stage selectively.
- **The plan content is authoritative.** Decisions in §1 are final — do not re-litigate. Sketches in §3 are deliberate, not drafts to improve. If something is genuinely unclear or self-contradictory, ask the user (Tony) rather than guess.
- **Tools**: use `Read` for files, `Edit` for surgical changes, `Write` only for new files or wholesale rewrites. Use `Bash` for `git mv`, `mkdir`, `rm`, `curl`, etc. Phase boundaries are natural commit candidates but defer all commit decisions to the user.
- **Resumption**: if your context is wiped mid-execution, §5 has the recovery checklist.

---

## 0. Goal

This worktree is being prepared for a future autonomous agent to spend several days iterating on **State Space Models** in the parameter-golf 16MB/10min/8×H100 challenge. The agent runs on MPS locally, ~5–25 min per experiment. The setup work happens NOW (this session). The SSM research happens LATER (their session 0 onwards).

Success criteria for the SETUP work (what you are doing now):
- Future agent reads `program.md` + `journal.md` and is oriented in <10 min
- Future agent has on-disk MPS-runnable Mamba reference (`mamba_minimal_model.py`) and a correctness oracle (`selective_scan_ref.py`)
- Inherited transformer-session artifacts are archived under `_archive_transformer/` (searchable, but not auto-loaded)
- One new skill (`noise-floor-sentinel`) added; no other process changes
- No new docs/ folder; no new top-level capital-letter files at the repo root beyond what this plan explicitly adds (just the rename of the long-named primer to `SSM_PRIMER.md`)

**Not your job**: deciding what the SSM agent should research, or pre-baking process bureaucracy. Set up the worktree and get out of the way. The existing structure (program.md, journal.md, summaries/, journals/, walks/, scratch/, records/, winners/, .claude/skills/) is robust — use it; don't invent new conventions.

---

## 1. Decisions (final — do not re-ask)

| # | Decision | Outcome |
|---|---|---|
| 1 | Vendor mamba-minimal? | **Yes — selective**. Vendor `mamba-minimal/model.py` (~300 LOC) and `selective_scan_ref` (~80 LOC). Other refs become curl-on-demand entries in `references/INDEX.md`. |
| 2 | Move primer + PAPERS.md + TECHNIQUES_INDEX.md to docs/? | **No**. No docs/ folder. Primer renamed in place at root (`SSM_PRIMER.md`); PAPERS.md and TECHNIQUES_INDEX.md edited in place additively (SSM section at top, transformer content kept below). |
| 3 | New `INHERITED_TRANSFORMER_NOTES.md` doc? | **No**. Archive-only. Agent greps `summaries/_archive_transformer/2026-04-25_overnight_session.md` if/when building a hybrid. |
| 4 | New `SSM_CHECKLIST.md` doc? | **No**. Distribute content into program.md (harness facts, math culture is already there) and references/INDEX.md (oracle usage). |
| 5 | New SSM pivot handoff summary? | **No**. Distribute content: G2 (primer-inconsistency) → journal.md Current threads; G3 (first-week recipe) → journal.md Open questions; targets/calibration → program.md "What you are NOT doing". |
| 6 | New `PROPOSED_PROGRAM_MD_CHANGES.md`? | **No**. The agent CAN edit program.md directly under the same trust model as journal.md. (Existing program.md "Permissions" section never restricted program.md itself.) |
| 7 | New `search_primer` skill? | **No**. One-line note in program.md "Reference materials": list sections with `grep -E '^##' SSM_PRIMER.md`, drill with `mdq`. |
| 8 | Lift `regression-sentinel` to its own skill? | **No**. Keep inline in program.md as today; the existing skill family is for mode-shifts and one-shot invocations, not periodic checklists. |
| 9 | New `noise-floor-sentinel` skill? | **Yes**. Architecture-family variance characterization. One-shot invocation, fits the existing skill family. |
| 10 | Archive prior session journals/summaries/walks? | **Yes** under `_archive_transformer/` subfolders. Update `search_journal` skill to exclude them by default; add explicit "search archives" mode. |
| 11 | Exit criterion for "give up on SSM"? | **No**. Contradicts the existing "NEVER STOP" section. The pivot-or-writeup decision is the human's, not the agent's. Existing pull-out / take-a-walk / hypothesis-discipline machinery handles dead-end recognition. |
| 12 | Special override mechanism for noise floor? | **No**. Keep program.md's existing thresholds as starting heuristics. After running noise-floor-sentinel, the agent applies rational discretion (the heuristic is "advance threshold = ≥3σ above measured floor"); they update journal.md Current threads with the measured value and adjusted thresholds. No two-source override layer. |
| 13 | New `derive-and-verify` skill (math discipline patterns)? | **Yes**. SSM math is meaningfully more error-prone than transformer math (continuous↔discrete transitions, mixed representations, recurrence amplifying errors over the sequence length). Existing pull-out covers *when* to do math; this skill covers *how* — patterns like worked tiny example, cite reference formula, recurrence-vs-convolution as free oracle, init invariants, degenerate cases. Lives as a skill (not in program.md) because it's invocation-driven: needed at the moment of derivation, not as ambient reference. |

---

## 2. Inventory

### Stay unchanged (do not touch)
- `train_gpt.py`, `train_gpt_mlx.py`, `data/`, `requirements.txt`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md` (the official challenge README)
- `records/`, `winners/`, `results.tsv`
- `await_steps.sh`, `new_experiment.sh`, `run_experiment.sh`
- `.claude/hooks/validate-subagent-plan.sh`, `.claude/settings.json`
- 6 of 8 existing skills: `launch-and-await`, `pull-out`, `zoom-in`, `take-a-walk`, `subagent-handoff`, `wrap-session`

### Edit in place (no move)
- `program.md` — sectional edits (~70% transfers verbatim; see §3 for exactly which sections change)
- `journal.md` — reset Current threads + Open questions; clear Stack of confirmed wins and Dead axes (let SSM agent rebuild)
- `PAPERS.md` — additive: SSM section at top, transformer content kept below as "Transformer/optimizer techniques retained (for hybrid composition)"
- `TECHNIQUES_INDEX.md` — additive: SSM section at top, existing records-validated transformer techniques kept below
- `.claude/skills/search_journal/SKILL.md` — light edit: default globs exclude `_archive_transformer/`; add explicit "search archives" mode
- `.claude/skills/promote/SKILL.md` — light edit: one sentence about SSM-family noise-floor caveat

### Renamed in place (no move)
- `State-space models for parameter golf a first-principles primer.md` → `SSM_PRIMER.md`

### Archived (move to `_archive_transformer/` subfolders)
- `summaries/2026-04-25_code_directions_session.md` → `summaries/_archive_transformer/`
- `summaries/2026-04-25_overnight_session.md` → `summaries/_archive_transformer/`
- `summaries/2026-04-25_width_curve_resumed.md` → `summaries/_archive_transformer/`
- `journals/2026-04-25_code_directions.md` → `journals/_archive_transformer/`
- `journals/2026-04-25_overnight_session.md` → `journals/_archive_transformer/`
- `journals/2026-04-25_width_curve_resumed.md` → `journals/_archive_transformer/`
- `walks/2026-04-25_1440.md` → `walks/_archive_transformer/`
- `walks/2026-04-25_1542.md` → `walks/_archive_transformer/`

### Deleted
- `scratch/brainstorm_big_directions.md`, `scratch/depth_recurrence_design.md`, `scratch/next_experiments.md`, `scratch/parking_lot.md` (gitignored, transformer-axis content). **Note**: `parking_lot.md` is gitignored and the parking-lot pattern is alive in the `zoom-in` skill — the SSM agent will re-create the file as needed; deletion now just clears stale transformer content.
- `SSM_SETUP_PLAN.md` at repo root (this doc; originally drafted at `scratch/SSM_SETUP_PLAN.md`, moved to root by the executing agent so progress commits track it. Deleted at end of execution in Phase F.)

### New
- `references/` — folder
- `references/INDEX.md` — index of vendored items + curl-on-demand external sources
- `references/mamba_minimal_model.py` — vendored from johnma2006/mamba-minimal (MIT)
- `references/selective_scan_ref.py` — vendored from state-spaces/mamba (Apache-2.0)
- `.claude/skills/noise-floor-sentinel/SKILL.md` — new skill (architecture-family variance characterization)
- `.claude/skills/derive-and-verify/SKILL.md` — new skill (math discipline patterns for SSM derivations)

Total new: 5 files + 1 directory.

---

## 3. Detailed content for each new/edited file

### 3.1 `program.md` — sectional edit (~70% transfers verbatim)

**Sections that transfer verbatim — DO NOT TOUCH** (read current program.md to confirm location):
- The "responsible and highly intellectual researcher" framing paragraphs (current lines 7-19)
- "Permissions" (current lines 56-71)
- "The experiment loop" steps 1-10 (current lines 72-94)
- "Auto-promote" (current lines 95-97)
- "Hypothesis discipline" (current lines 99-124)
- "Logging formats" → results.tsv schema and journal entry skeleton (current lines 126-164)
- "Heading craft" rules (current lines 156-164)
- "Noise floor" sub-section thresholds (current lines 165-171) — body kept verbatim; ONE sentence added at the top of the section as a pointer (see edit (g) below). This keeps the framework intact while explicitly flagging that program.md numbers are starting heuristics, not authoritative for SSM.
- "Soft constraints" (current lines 173-177)
- "Regression sentinel" (current lines 179-189) — keep verbatim, do NOT lift to a skill
- "Running experiments" (current lines 191-193)
- "Subagent for code edits" (current lines 195-197)
- "Wrapping a session" (current lines 199-201)
- "NEVER STOP" (current lines 209-221)
- "When the human returns and explicitly asks you to STOP" (current lines 223-225)

**Sections to edit**:

#### (a) Replace title block + "From Previous Session" block (current lines 1-31)

Replace the existing top-of-file (everything from `# program.md — Parameter Golf Autoresearch` through the end of the "From Previous Session" section, ending just before `## Reference baseline`) with:

```
# program.md — Parameter Golf SSM Autoresearch

You are an autonomous research agent exploring **State Space Models** in the parameter-golf 16 MB / 10 min / 8×H100 challenge. Goal: minimize validation bits-per-byte (`val_bpb`) on the 200-step MPS smoke locally; the human evaluates final candidates on 8×H100s for 20k-step training. You iterate on `train_gpt.py` (forked per experiment). Your job is **directional exploration** of an architecture family that has not been competitively explored in this challenge.

[Then keep verbatim the existing "responsible and highly intellectual researcher" paragraphs from current lines 7-19, including the "bold + rigorous", the "stuck/circling", the mode-shift paragraph, the take-a-walk paragraph, and "Good luck!".]

You run autonomously. The human is asleep or away. You promote your own wins, journal your own findings, and continue until manually stopped.

## What you are NOT doing

- Not optimizing the existing transformer config. The previous session got val_bpb 2.087 (exp 0062, K=3 L=3 + SwiGLU mlp=8) on the MPS smoke; that is your *comparison anchor*, not a starting fork to tune.
- Not trying to set leaderboard SOTA. Realistic targets at our regime per `SSM_PRIMER.md` §4.7: beat naive baseline (val_bpb < 2.521 on this MPS smoke), match transformer best (< 2.087), produce an honest non-record submission for OpenAI's wishlist track. Beating 0062 substantively is aspirational — the primer's main body estimates <2% probability in any short budget; its critique section disagrees (~30-40% probability of an "interesting result" that beats 1.18 BPB on H100). Both are research opinions; see journal.md Current threads.
- Not running with assumed thresholds. The previous session's noise floor (~0.0024 cross-seed for stable transformer configs) does not auto-transfer. Mamba's documented sharp LR cliffs (primer §4.2) make freak single-seed runs more likely; SSM noise floor is likely different. Characterize via the `noise-floor-sentinel` skill on your first stable SSM block.
- **Not promoting before noise-floor-sentinel for the architecture family**. Until the sentinel completes for an SSM family, every win is `status=keep` only — never invoke `promote`. The previous transformer session's documented anti-pattern (single-seed direct-promote-zone wins piling up before cross-seed confirms) is more dangerous in the SSM regime where Mamba's LR cliffs make freak-good first-seed runs more likely. This is operational rule, not suggestion.

The deliverable is the work + the writeup. Even without a leaderboard win, an honest characterization of what was tried, what failed, and *why* (with derivations and measurements) is a valuable contribution to OpenAI's wishlist track — and is the actual goal here. Don't grind on the bpb axis past the point where the writeup is the higher-value output.
```

#### (b) Replace "Reference baseline" section (current lines 33-39)

Replace with:

```
## Reference baseline

The harness anchor is **experiment 0001_baseline_repro** in `results.tsv`, val_bpb 2.5212 post-quant, 6.907 MB, 200 steps. Every regression check and Δ-comparison goes against that row.

The previous session's transformer best (val_bpb 2.08687, exp 0062, K=3 L=3 + SwiGLU mlp=8, path `winners/2026-04-25_recur_3x3_swiglu_mlp8/`) is a comparison anchor only. **Do not inherit the architecture** (recurrence + SwiGLU MLP=8) — that defeats the SSM exploration goal. **Do inherit the schedule/optimizer/init defaults** — they're architecture-independent and tuned for the 200-step MPS regime. See journal.md Current threads "Starting env.sh for SSM experiments" for the specific values (WARMDOWN_ITERS=300, LR_WARMUP_STEPS=30, TIED_EMBED_INIT_STD=0.05, MUON_BACKEND_STEPS=15, TRAIN_BATCH_TOKENS=24576, MATRIX_LR=0.045). Running an SSM block on canonical defaults (warmdown=1200, warmup=0, batch=8192, init=0.005, muon_steps=5) confounds architecture signal with under-training. Exception: regression-sentinel uses canonical, since its job is harness-drift detection against 0001_baseline_repro. For hybrid-composition details beyond env.sh, grep the archived summary for "Recommendations" or "Stack of confirmed wins".

MPS characteristics:
- Transformer step: ~1.2 s/step → ~5 min per experiment, ~80 overnight.
- SSM step time depends on block class **[CONJECTURE]**: S4D-Lin (FFT-conv, no selectivity) likely close to transformer speed; Mamba-1 sequential `selective_scan` ~3-6× slower per primer §4.1 → ~15-25 min per experiment. Characterize empirically in your first 2-3 experiments before committing to overnight schedules.
- **Do NOT set `VAL_TOKENS=0`.** The full-val eval is called twice (pre-quant + post-int8-quant) and each pass is ~30+ min on MPS — total runtime ~60–120 min per experiment, killing throughput. Stick with the cap. The 16K-cap sample is enough for ranking at the 0.010 noise floor (sampling error cancels in same-seed Δ comparisons).
```

#### (c) Insert new section AFTER "Reference baseline", BEFORE "Setup (every session)"

```
## SSM-specific harness facts

- **`CONTROL_TENSOR_NAME_PATTERNS` is env-driven** (train_gpt.py line 304-311). It does triple-duty: matched tensors are (1) kept fp32 during training (`restore_low_dim_params_to_fp32`, line 532-537), (2) kept fp32 at quant export (`keep_float_tensor`, line 329-335), and (3) routed to AdamW instead of Muon (line 901). `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS` (line 312-319) defaults to mirror `CONTROL_TENSOR_NAME_PATTERNS`, so extending one extends both unless you override. Mamba's official README says "SSMs are sensitive to their recurrent dynamics — use fp32 for parameters." **Append to the canonical default, do not retype from memory** — read train_gpt.py line 308 for the current canonical value (singular AND plural forms matter; substring match is exact). Example env.sh that appends correctly:
  ```
  # Canonical default (read from train_gpt.py:308) + SSM extensions:
  export CONTROL_TENSOR_NAME_PATTERNS="attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,A_log,D,dt_bias,dt_proj,delta_bias"
  ```
  Verify wiring at first forward by printing the names matched by `restore_low_dim_params_to_fp32`. Verify at quant export by checking those tensors land in `passthrough` not `quantized` of the int8 obj.
- **Numel cap on the keep-float pathway**: `INT8_KEEP_FLOAT_MAX_NUMEL = 65_536` per tensor (line 320, **hardcoded — not env-driven**). At d_inner=512: d_state ≤ 128 (→ 65536) fits; d_state ≥ 129 exceeds and will be int8-quantized regardless of `CONTROL_TENSOR_NAME_PATTERNS`. If you need d_state > 128 at d_inner=512, raise the cap inside your **experiment-folder** `train_gpt.py` copy (NOT at root — root is canonical and locked) and document the override in plan.md so future agents see why it's there.
- **Cap math for SSM differs from transformer math**. Tensors matched by `CONTROL_TENSOR_NAME_PATTERNS` are kept fp32 (4 bytes/elem) at quant export, not int8 (1 byte). Derive the *post-quant* artifact size in `scratch/` before training any block whose fp32-protected mass is non-trivial. A Mamba block with `d_inner=512, d_state=64` and the recommended A_log/D/dt_bias/delta_bias protected adds ~0.13 MB fp32 vs ~0.03 MB int8 per layer — small per-block but compounds over 9 layers.
- **MPS reality**: pure-PyTorch `selective_scan` is sequential on MPS. You CANNOT install `mamba-ssm`, `causal-conv1d`, or Triton — all CUDA-only. Use vendored `references/mamba_minimal_model.py` for Mamba-1 reference; use `references/selective_scan_ref.py` as your correctness oracle (numerical agreement on a small fixed input via `torch.allclose(your_out, ref_out, atol=1e-5, rtol=1e-4)` before you trust any custom scan in an experiment). The recurrence amplifies bugs over the sequence length — a step-1 anomaly that would be a curiosity in a transformer is often a smoking gun in an SSM.
- **Late-NaN gate is non-optional for SSMs**. After the standard step-1-to-10 trajectory gate (`launch-and-await` skill), run an additional `await_steps.sh ... 100` block before treating an SSM run as healthy. Mamba-family late instability around step 50-150 is a documented failure mode (primer §4.2: sharp LR cliffs); a clean step-10 trajectory is necessary but not sufficient.
- **Throughput cost is real**. PR #831 calibrated on H100: at 83 ms/step, each 1 ms overhead costs ~7 optimizer steps; each step improves BPB by ~0.001; therefore any technique must improve BPB by ≥0.007 per ms of overhead. The principle transfers; the constant doesn't (MPS math differs; your model differs). Form your own threshold after pairs of experiments where one differs only in step time. Until then treat 0.007/ms as a sanity-check ballpark, not a promotion gate. The harness already tracks `step_avg_ms` in results.tsv.
- **Tokenizer is locked at sp1024**. Cannot upgrade to sp4096/sp8192/Scylla. The H100 records below 1.10 BPB mostly use larger vocabs; do not chase those numbers.
```

#### (d) Edit "Setup (every session)" (current lines 41-48)

Replace the existing 6-step list with:

```
## Setup (every session)

1. Read this file in full.
2. Read `journal.md` (Current threads first, then recent entries newest-first) and all top-level files in `summaries/`. `_archive_transformer/` is searchable on demand but NOT default reading — the `search_journal` skill carries the patterns.
3. Skim `results.tsv`. Bottom rows are SSM experiments; older rows are transformer history.
4. `git log --oneline -10` for canonical state.
5. `date` to anchor in time. Note any wrap-time the human gave at the top of your first journal entry.
6. **Session 0 only**: read `SSM_PRIMER.md` end-to-end once (~9.7k words, ~30k tokens). On subsequent sessions, do NOT re-read it — drill by section: `mdq '# "<keyword>"' SSM_PRIMER.md`. List sections: `grep -E '^##' SSM_PRIMER.md`.
7. **Recommended on first session in this worktree**: run a regression sentinel (slug `regression_check_001`, no env-var changes) to verify the harness still bit-reproduces 0001 (val_bpb 2.5212 ± 0.005). Cheap insurance — ~5 min — before pouring compute into novel SSM work. Skip if you have a specific reason to (e.g., already verified).
```

#### (e) Replace "Reference materials (browse selectively)" section (current lines 203-207)

Replace with:

```
## Reference materials

- `SSM_PRIMER.md` — the rigorous SSM primer (~9.7k words). Read end-to-end **once** in session 0; subsequent sessions, drill by section: `mdq '# "<keyword>"' SSM_PRIMER.md`. List sections: `grep -E '^##' SSM_PRIMER.md`. **The primer is internally inconsistent in places** — the main body argues SSM-on-Parameter-Golf is "almost certainly wrong"; the "Another agent's feedback to this document" section disagrees on three points (quantization fragility, recall remedy via BigramHash, probability of an interesting result). Both are research opinions; verify with measurement and log empirical updates as `Empirical update to primer §X: ...` in journal.md.
- `PAPERS.md` — curated arxiv reading list. SSM-focused at top; transformer/optimizer techniques retained below for hybrid composition. Fetch with `curl https://arxiv.org/pdf/<id>`.
- `TECHNIQUES_INDEX.md` — SSM technique families at top; transformer-records summary below for hybrid composition.
- `references/INDEX.md` — vendored mamba-minimal + selective_scan_ref + curl-on-demand pointers. Vendored code in `references/` is licensed for adaptation (with attribution headers preserved); adapt freely.
- `records/` — transformer leaderboard records. Read for *categories of techniques* (BigramHash, GPTQ, EMA, depth recurrence) — **do not copy code**. These are active leaderboard submissions under their own licenses; plagiarism defeats the point.
- `winners/` — prior session's transformer wins. Snapshot of exp 0062 (val_bpb 2.087) is `winners/2026-04-25_recur_3x3_swiglu_mlp8/`. Read for context, not as starting forks (unless explicitly building a hybrid).
```

#### (g) Add one sentence at the TOP of the "Noise floor" sub-section (current line 165, just under the `### Noise floor (200-step smoke, VAL_TOKENS=16384)` heading)

The bullets below transfer verbatim. Add this single sentence between the heading and the first bullet:

> These thresholds were calibrated to the transformer noise floor (~0.0024 cross-seed); they are **starting heuristics for SSM work, not authoritative**. After `noise-floor-sentinel` runs for an architecture family, journal.md Current threads holds the σ-anchored thresholds for that family — defer to those.

That's the full set of program.md edits. **No other section changes.** Verify after editing: line count target 280-310. (Math discipline patterns live in the new `derive-and-verify` skill — see §3.10 — not in program.md.)

### 3.2 `journal.md` — full reset of Current threads + Open questions

Replace the entire file content from the top through the end of the "Open questions" section (current lines 1-54) with the block below. Leave the `## Entries (newest first)` marker at the bottom (already present in current line 56) untouched.

```markdown
# Journal

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Inherited transformer best (comparison anchor only)**: exp 0062 val_bpb 2.08687, K=3 L=3 + SwiGLU(mlp=8). Path: `winners/2026-04-25_recur_3x3_swiglu_mlp8/`. Reference for "what an optimized transformer at our regime achieves." **Do not inherit the architecture** (recurrence + SwiGLU MLP=8) — that defeats the SSM exploration goal. **Do inherit the schedule/optimizer/init defaults** below (architecture-independent, [transfer:high]). For full hybrid composition details, grep `summaries/_archive_transformer/2026-04-25_overnight_session.md` for "Recommendations" or "Stack of confirmed wins".

- **Starting env.sh for SSM experiments** (architecture-independent transformer wins, [transfer:high] in archive). Set these in your env.sh for any SSM experiment to avoid running on canonical defaults that under-train at 200 steps:
  ```
  WARMDOWN_ITERS=300
  LR_WARMUP_STEPS=30
  TIED_EMBED_INIT_STD=0.05
  MUON_BACKEND_STEPS=15
  TRAIN_BATCH_TOKENS=24576
  MATRIX_LR=0.045
  ```
  Canonical (warmdown=1200, warmup=0, batch=8192, init=0.005, muon_steps=5) is the pre-fix regime; running an SSM block on canonical confounds architecture signal with under-training. **Exception: regression-sentinel uses canonical defaults** — its job is harness-drift detection against 0001_baseline_repro, which was recorded on canonical.
- **SSM-family noise floor: UNCHARACTERIZED**. The transformer floor of ~0.0024 cross-seed does not auto-transfer. Run the `noise-floor-sentinel` skill on your first stable SSM block. **Until the sentinel completes for an architecture family, do NOT invoke `promote` on any SSM-family experiment in that family — treat any apparent win as informational only.** Mamba's sharp LR cliffs (primer §4.2) make freak-good first-seed runs more likely; the previous transformer session's documented anti-pattern (single-seed direct-promote-zone wins piling up before cross-seed confirms — see `summaries/_archive_transformer/2026-04-25_overnight_session.md` "methodology debt") is the exact failure mode this guardrail prevents. Update this bullet with measured σ and adjusted thresholds when sentinel completes — e.g., "S4D-Lin noise floor: σ=X measured 2026-04-26 exp NNNN-NNNN; advance threshold Δ ≥ 3σ; judgment-call window [2σ, 3σ]."
- **Primer is internally inconsistent**: main body argues SSM is "almost certainly wrong" for parameter golf; the "Another agent's feedback" section disagrees on (a) whether to quantize the SSM (the `CONTROL_TENSOR_NAME_PATTERNS` env var makes "don't quantize" one line), (b) whether BigramHash closes the recall gap, (c) the probability of an interesting result. Treat both as research opinions; verify empirical claims with measurement; log empirical updates as `Empirical update to primer §X: ...` in entries.
- **MPS reality** [CONJECTURE]: ~5 min per experiment for transformer-speed blocks (S4D-Lin FFT-conv likely lands here); ~15-25 min for sequential `selective_scan` (Mamba-1). Characterize in your first 2-3 experiments. CUDA kernels (mamba-ssm, causal-conv1d, Triton) unavailable — use vendored `references/mamba_minimal_model.py`.
- **Tokenizer is locked at sp1024**.

## Stack of confirmed wins (cumulative path canonical → current best)

(empty — populated as SSM wins land. Inherited transformer wins are in `summaries/_archive_transformer/2026-04-25_overnight_session.md`.)

## Dead axes (verified — don't re-test without changing other levers)

(empty — populated as SSM dead axes are verified. Transformer-axis dead-axes from prior session are NOT auto-transferred to SSM regime; verify before assuming.)

## Open questions (next session priorities)

A starting recipe based on primer §4.7 + the primer's "Another agent's feedback" 6-item ranked list. **One researcher's recipe — diverge with reason and document why.** This is a starting menu, not a binding sequence.

1. **Get an SSM block running on Mac iteration loop**. Vendored `references/mamba_minimal_model.py` is the starting point. Goal: forward pass clean, no NaN, get *any* val_bpb. Target: < 2.521 (beats baseline). Primer estimate: half a day.

   **Before training, derive in `scratch/`** (extension of program.md "Measurement over belief" + pull-out's "compute parameter count, sketch the math"). The recurrence amplifies math errors over the sequence length, so untested derivations become smoking guns:
   - **Param count** of the block as a function of `d_inner, d_state, d_conv`. Then *post-quant* artifact size — anything in `CONTROL_TENSOR_NAME_PATTERNS` stays fp32 (4×), so the cap math differs from transformer math (see program.md "SSM-specific harness facts"). Confirm artifact stays under cap.
   - **Eigenvalue placement** of `Ā = exp(ΔA)` post-discretization. For stability you need `|λ| ≤ 1` for all eigenvalues; the standard `A = -exp(A_log)` parameterization gives this for free, but verify with a small print on init.
   - **Kernel formula** for LTI blocks (S4, S4D): `K = (CB̄, CĀB̄, CĀ²B̄, ..., CĀ^(L-1)B̄)`. Compute symbolically on a tiny case and `torch.allclose` against your conv kernel before training.
   - **Numerical agreement against `selective_scan_ref`** for any selective (Mamba-family) scan. Build a small fixed input (e.g. B=2, L=64, D=16, N=8 — adjust to your block); run both your scan and the oracle; `torch.allclose(out, ref, atol=1e-5, rtol=1e-4)` should be True. Debug before training.

   These are the concrete derivations to do for THIS first SSM block. **For HOW to do math well — patterns like worked tiny example, recurrence-vs-convolution as oracle, init invariants, degenerate cases — invoke the `derive-and-verify` skill.** A failed derivation upstream is much cheaper to fix than a "why is this NaN at step 50" debug downstream.
2. **Pick discretization wisely**. Primer suggests S4D-Lin (LTI, ZOH-discretized, two lines of code, debug-friendliest) before Mamba's selective scan. Diverge with reason.
3. **Decide what NOT to quantize**. If the SSM is a small fraction of total params, `CONTROL_TENSOR_NAME_PATTERNS` keeps it fp32 (program.md "SSM-specific harness facts"). The primer's main body and critique disagree on quantization-hostility — measure with vs without protection on your first stable config to settle it for *your* architecture.
4. **Run noise-floor-sentinel** on the working config from (1). ~3 experiments, characterizes architecture-family variance. Required before treating any subsequent Δ as signal.
5. **S4D vs Mamba-1 vs Mamba-2/SSD bake-off** at the same single-replaced-layer position. Determines which selective family to invest in. Each is one experiment with a different vendored block.
6. **Don't sweep LR exhaustively**. Primer suggests 3 points {0.005, 0.01, 0.02} based on Mamba's documented sharp LR cliffs (primer §4.2). Diverge if measurement shows the cliff isn't sharp at your scale.
7. **BigramHash recall compensation** if pure SSM lags on val_bpb (primer §4.5 "Zoology" lesson — 82% of the SSM↔attention perplexity gap is associative recall; BigramHash is record-validated, ~30 lines, subagent task).
8. **Hymba-lite parallel attn+SSM heads** (primer §4.6) is *one* option among several; note that on-leaderboard Hymba (1.1828) and S4D-Lin hybrid (1.1682) both lost to contemporaneous transformers by 0.06–0.10 BPB. Cautious-known-losing has low upside. Less-tried families (GLA, Hyena, RetNet, GateLoop — see PAPERS.md) deserve weight.
9. **Quant interaction**: at first stable SSM config, measure quant_tax. Per primer §4.4, expect amplification; the fp32-protect knob limits damage but may not eliminate it.

### Open question — depth recurrence transfer (untested at SSM regime)
The previous transformer session found a depth-recurrence win (K=3 L=3 looped, +0.0055 vs flat 9L mlp=4 baseline at our 200-step MPS regime). Issue #140 commentary on the Hymba submission stated *"SSM makes each layer more powerful → 7L beats deeper pure transformers at same step budget."* Whether the depth-recurrence instinct transfers to SSM blocks is **untested at our regime**. If you build a hybrid containing both block types, consider testing both directions (looped K=3 vs flat 9L with stronger SSM) rather than assuming Issue #140's framing applies. Log empirical findings as `Empirical update to depth-recurrence question: ...`.

### Open question — scale deception (one documented failure mode)
PR #1227 (SSM hybrid) improved CE 18% at d=192 but regressed BPB 2.7% at d=512. Issue #140 calls this "scale deception" — one PR's failure; whether it generalizes to your architecture is your job to verify. Reasonable practice: if you tune at smaller scale to iterate faster, re-test at the operating scale (d=512) before promotion. Principle transfers; magnitude isn't a law.

## Entries (newest first)
```

(The `## Entries (newest first)` marker is the existing line 56 of journal.md — don't double up; just leave it as-is at the end.)

### 3.3 `references/INDEX.md` (new — full content)

```markdown
# References — index of code resources for SSM work

## Vendored (load-bearing — read directly in editor)

| File | What it is | When to use |
|---|---|---|
| `mamba_minimal_model.py` | johnma2006/mamba-minimal `model.py`, ~300 LOC pure PyTorch. Numerically equivalent to official Mamba forward/backward. Sequential `selective_scan` (slow but correct on MPS). License: MIT. | Building any Mamba-1 block in this worktree. Adapt rather than reimplement from arxiv. |
| `selective_scan_ref.py` | Official `selective_scan_ref` from state-spaces/mamba, ~80 LOC. License: Apache-2.0. | **Correctness oracle.** Any custom selective-scan you write must pass numerical agreement with this on a small fixed input before you trust it in an experiment. |

### Oracle usage protocol
Before integrating any custom selective-scan into train_gpt.py:
1. Build a small fixed input: e.g., B=2, L=64, D=16, N=8.
2. Run both your scan and `selective_scan_ref` on the same input + same params.
3. `torch.allclose(your_out, ref_out, atol=1e-5, rtol=1e-4)` should be True.
4. If it fails, your scan is wrong. Debug before training. The recurrence amplifies bugs over the sequence length; a step-1 anomaly that would be a curiosity in a transformer is often a smoking gun in an SSM.

## Curl-on-demand (don't preemptively download)

Fetch when needed; vendoring everything would clutter the worktree.

### Mamba implementations
- mamba-2 minimal (tommyip): `git clone --depth=1 https://github.com/tommyip/mamba2-minimal references/_external/mamba2-minimal`
- mamba.py / MambaPy (alxndrTL — includes parallel associative `pscan`, MPS-compatible): `git clone --depth=1 https://github.com/alxndrTL/mamba.py references/_external/mambapy`. Suggested vendoring criterion (not a rule): vendor only when slow-but-correct mamba-minimal works AND step time is the experimental bottleneck (e.g. >30 min per experiment).
- The Annotated Mamba (Sasha Rush, Triton): https://srush.github.io/annotated-mamba/hard.html
- `state-spaces/mamba/mamba_ssm/modules/ssd_minimal.py` — Listing 1 of the Mamba-2 paper, ~30-line pure-PyTorch chunk-decomposition (matmul-only, MPS-compatible). Trade-off vs Mamba-1: scalar-per-time A_t (less expressive per state) but allows much larger N (state dim 64-256 vs Mamba-1's 16).

### S4 / S4D
- The Annotated S4 (best pedagogy, JAX): https://srush.github.io/annotated-s4/
- The Annotated S4D: https://srush.github.io/annotated-s4/s4d
- Official state-spaces/s4 (Apache-2.0): `git clone --depth=1 https://github.com/state-spaces/s4 references/_external/s4`

### Blogs (read in browser via curl + html-to-text if useful)
- Maarten Grootendorst — Visual Guide to Mamba: https://www.maartengrootendorst.com/blog/mamba/ (best intuition)
- Tri Dao — Mamba-2 Parts I-IV: https://tridao.me/blog/2024/mamba2-part1-model/
- Goomba Lab — Mamba-2 Parts I-IV: https://goombalab.github.io/blog/2024/mamba2-part1-model/
- Goomba Lab — Tradeoffs of SSMs and Transformers (2025): https://goombalab.github.io/blog/2025/tradeoffs/
- Hazy Research — Zoology / MQAR: https://hazyresearch.stanford.edu/blog/2023-12-11-zoology0-intro

### Papers
Use `PAPERS.md` as the curated index. Fetch with `curl https://arxiv.org/pdf/<id>` then process locally.

## Notes on .gitignore
`references/_external/` is pre-staged in `.gitignore` (added during setup) so curl-on-demand vendoring of upstream repos doesn't accidentally commit other people's code. If you fetch a substantial repo elsewhere, follow the same pattern (add to .gitignore before cloning).
```

### 3.4 `references/mamba_minimal_model.py` and `references/selective_scan_ref.py` (vendored)

For each file:
1. Fetch the upstream content.
2. **Add a header comment** at the very top with: source URL, upstream commit SHA (from the fetch response), license name, fetch date in ISO format, and (for selective_scan_ref) any required Apache-2.0 NOTICE attribution.
3. Verify the file parses: `python3 -c "import ast; ast.parse(open('references/<file>.py').read())"`.
4. For `selective_scan_ref.py`, verify it's standalone — only `torch` imports, no internal `mamba_ssm.*` dependencies. If it has internal imports, simplify to standalone before committing (the official version is already nearly standalone; just inline any small helper).

**Apache-2.0 obligation for `selective_scan_ref.py`**: before committing, check the upstream `state-spaces/mamba` repo for a `NOTICE` file or copyright headers. Apache-2.0 §4(d) requires preserving any NOTICE content. If present, include the relevant copyright/notice text in the file header. Header template:

```python
# Vendored from state-spaces/mamba
# Source: https://github.com/state-spaces/mamba/blob/<commit-sha>/<path>
# Commit: <sha>
# Fetched: 2026-04-26
# License: Apache-2.0 (see https://github.com/state-spaces/mamba/blob/<sha>/LICENSE)
# Copyright <year> <upstream copyright holder>
# [If upstream NOTICE file exists, include its content verbatim below]
```

For `mamba_minimal_model.py` (MIT), simpler header:

```python
# Vendored from johnma2006/mamba-minimal
# Source: https://github.com/johnma2006/mamba-minimal/blob/<sha>/model.py
# Commit: <sha>
# Fetched: 2026-04-26
# License: MIT
# Copyright (c) 2023 John Ma
```

### 3.5 `.claude/skills/noise-floor-sentinel/SKILL.md` (new — full content)

```markdown
---
name: noise-floor-sentinel
description: Invoke before treating any Δ as signal on a new SSM architecture family. Runs 3 same-config seeds to characterize the empirical noise floor for THIS architecture class. Distinct from program.md "Regression sentinel" (harness-drift check) — this is architecture-family variance characterization, which Mamba's LR cliffs (primer §4.2) make even more important than for transformers.
---

# Noise-Floor Sentinel

## Hard rule
**No SSM-family experiment is promoted before this skill completes for that family.** Until you have the family's measured σ, treat any apparent win as informational only — do not invoke `promote`. The previous transformer session piled up single-seed direct-promote-zone wins before cross-seed confirms (see archived methodology-debt entries); Mamba's sharp LR cliffs make freak-good first-seed runs more likely than in the transformer regime, so the same anti-pattern is more dangerous here.

## When to invoke
Concrete gate: invoke only after a config that meets ALL of:
- Forward pass produces no NaN/Inf
- Step-1 train_loss ≈ ln(vocab) (≈ 6.93 at sp1024); step 2 within ~2× of step 1; monotonic descent thereafter (per `launch-and-await` trajectory checks, plus the SSM late-NaN gate at step 100)
- val_bpb_post_quant < 2.521 (beats the naive baseline 0001) — characterizing variance of a fundamentally broken config wastes 3× experiments and the σ doesn't generalize

Then invoke:
- After your first stable SSM block in a new architecture family
- When you change the SSM block class (S4D → Mamba-1 → Mamba-2/SSD)
- When you change the SSM placement (single layer replaced → all layers SSM → parallel attn+SSM heads)
- NOT for hyperparameter changes within an already-characterized config (those are signal, not new variance)

## Why it matters
The previous transformer session established a 0.0024–0.003 cross-seed noise floor; program.md's promote thresholds (Δ ≥ +0.010 advance, [+0.005, +0.010] judgment-call) are tuned to it — that's roughly [2σ, 4σ] empirically. SSMs may have a different floor. Per primer §4.2, Mamba has sharp LR cliffs and is more sensitive than transformers to init/hyperparameter perturbation. A "freak good" or "freak bad" single-seed run is more likely. Without knowing the floor, you'll either promote noise or discard real signal.

## Procedure
1. Pick a stable working config of the new architecture family (the one that meets the concrete gate above — it will be your "baseline" for this family). Note its experiment id (e.g., `0073`).
2. Run 3 experiments forking from that stable config — **NOT from canonical**. Use `./new_experiment.sh <slug> <stable_ssm_exp_id>` so the fork inherits the SSM block code, env.sh, and any train_gpt.py edits. Then change ONLY the SEED:
   ```
   ./new_experiment.sh <family>_sentinel_seed42 <stable_ssm_exp_id>
   # in the new experiment's env.sh, set SEED=42 (override the parent's seed)
   # fill plan.md (Question="noise-floor sentinel <family>", Hypothesis="cross-seed Δ within ±2σ", Change="SEED only", Disconfirming="bimodal or wide spread")
   cd experiments/NNNN_<family>_sentinel_seed42 && ../../run_experiment.sh
   ```
   Repeat for 2 more seeds. **Pick seeds disjoint from your stable config's seed.** If the stable config used the canonical `SEED=1337`, run `SEED=42`, `SEED=2024`, `SEED=31337`. (`SEED=2024` and `SEED=31337` haven't been used in this codebase before; they're plain torch RNG seeds and should work — a one-line sanity check that step-1 train_loss differs across the three is cheap insurance against accidentally running duplicates.)
3. Each experiment: status `sentinel`, description "noise-floor sentinel for <family> seed <N>".
4. After all 3 complete, journal with the entry header `## YYYY-MM-DD · <family> noise floor (3 seeds)`:
   - Three val_bpb_post_quant values
   - Mean and standard deviation
   - Cross-seed Δ (max - min)
   - Verdict: noise floor for this family is σ ≈ X
5. **If the three seeds are bimodal** (one freak run + two tight, or two-and-two split — i.e., the spread looks like LR-cliff behavior rather than a smooth gaussian), run a fourth seed before concluding. Mamba's LR cliffs (primer §4.2) make this regime exactly where a 3-seed sample is too small. The fourth seed disambiguates "freak event in tail" vs "true bimodal config."
6. Update journal.md Current threads with the new floor, using thresholds anchored to the measured σ. Use the same formula structure as the transformer rule (judgment_low = advance/2, the empirical 0.5 ratio):
   - **Advance**: Δ ≥ Y where Y is the family's advance threshold. Suggested Y = 3σ (standard "p < 0.003" significance heuristic). Y = 4σ mirrors the transformer's empirical anchor (transformer's advance=0.010 ≈ 4.17σ at σ=0.0024); pick whichever you can defend.
   - **Judgment-call** (re-run with an extra seed; advance only if Δ holds): Δ ∈ [Y/2, Y]
   - **Noise**: Δ < Y/2
   Format: `**<family> noise floor**: σ=X measured exp NNNN-NNNN; advance Δ ≥ Y=3σ=<value>; judgment [Y/2, Y] = [<value>, <value>].`

## Cost
3 experiments (4 if bimodal). Cost depends on architecture family:
- S4D-Lin (FFT-conv, transformer-speed step time): ~15 min total for 3 seeds; ~20 min if bimodal
- Mamba-1 sequential `selective_scan` (3-6× slower per primer §4.1): ~45-75 min for 3 seeds; ~60-100 min if bimodal

Required investment, not optional overhead.

## Distinct from regression-sentinel
| Sentinel | Asks | Cadence |
|---|---|---|
| Regression sentinel (program.md "Regression sentinel") | Is the harness still bit-reproducing 0001? | Every 10 experiments |
| Noise-floor sentinel (this skill) | What is THIS architecture family's cross-seed variance? | Once per architecture-family change |

## After the sentinel
With the floor measured, return to the experiment loop with calibrated thresholds. If the SSM noise floor is e.g. σ=0.008 (3× the transformer floor), then promote-skill `Δ ≥ +0.010` becomes `Δ ≥ +0.024` for THIS family. Document in journal.md Current threads.
```

### 3.6 `.claude/skills/promote/SKILL.md` (light edit)

In the existing "Decide if it's actually a win" section (current SKILL.md line 10), add this paragraph between the section header and the existing bulleted Δ rules:

> **Hard rule for SSM-family experiments**: do NOT promote any SSM-family win before the `noise-floor-sentinel` skill has completed for that architecture family. Treat apparent wins as informational only until you have the family's measured σ. The thresholds below were calibrated to the transformer noise floor (~0.0024 cross-seed; advance=0.010 ≈ 4.17σ); for SSM, replace them with σ-anchored thresholds using the same shape (advance Δ ≥ Y where Y = ≥3σ; judgment-call [Y/2, Y]; noise <Y/2) once the sentinel completes. The judgment-low = advance/2 ratio matches the transformer rule. Document the adjusted Y and σ in journal.md Current threads.

No other changes to this skill.

### 3.7 `.claude/skills/search_journal/SKILL.md` (light edit)

Replace the existing three-mode body (current lines 6-28) with:

```markdown
Three modes for finding past work. Pick the one that matches what you know:

- **Browse** — TOC across active sessions:
  ```
  grep "^## " journal.md journals/[!_]*.md summaries/[!_]*.md
  ```
  The `[!_]` glob skips files starting with underscore — i.e., excludes `_archive_transformer/` content. Default browse focuses on the current research arc.

- **Search** — topic across active sessions:
  ```
  grep -i "<topic>" journal.md journals/[!_]*.md summaries/[!_]*.md
  ```
  Full content. Use this when the topic might be discussed inside an entry whose heading doesn't name it.

- **Drill** — full content of a known heading:
  ```
  mdq '# "<keyword>"' journal.md journals/[!_]*.md
  ```
  `#` matches any heading at any depth (single hash regardless of `##` / `###`). Quoted substring match.

If a query returns nothing or too much, switch modes — usually browse/search → drill, narrowing each time.

## Searching archives explicitly (opt-in)

Archived transformer-session journals/summaries/walks live in `journals/_archive_transformer/`, `summaries/_archive_transformer/`, `walks/_archive_transformer/`. They contain ~65 experiments of transformer-axis history. Default search excludes them. To search archives:

```
# Browse archive headings:
grep "^## " journals/_archive_transformer/*.md summaries/_archive_transformer/*.md

# Search archive contents:
grep -i "<topic>" journals/_archive_transformer/*.md summaries/_archive_transformer/*.md
```

When in doubt about whether an archive search is warranted: usually it isn't. The transferable lessons most relevant to a hybrid (TIED_EMBED_INIT_STD=0.05, MUON_BACKEND_STEPS=15, batch=24k+matrix_lr=0.045, etc.) live in `summaries/_archive_transformer/2026-04-25_overnight_session.md`.
```

### 3.8 `PAPERS.md` (additive edit at root)

**Read the current file first** (`Read PAPERS.md`) to see the existing transformer/optimizer content. Then prepend (above all existing content) this new section, and wrap the existing content under a new header `## Transformer/optimizer techniques retained (for hybrid composition)`.

New content to prepend:

```markdown
# Curated arxiv reading list

## State Space Models (current focus)

### Foundational
- HiPPO: Recurrent Memory with Optimal Polynomial Projections — arxiv:2008.07669 (Voelker, Gu, Re)
- S4: Efficiently Modeling Long Sequences with Structured State Spaces — arxiv:2111.00396 (Gu, Goel, Re)
- DSS: Diagonal State Spaces are as Effective as Structured State Spaces — arxiv:2203.14343 (Gupta, Gu, Berant)
- S4D: On the Parameterization and Initialization of Diagonal State Space Models — arxiv:2206.11893 (Gu, Goel, Gupta, Re)

### Selective and gated
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces — arxiv:2312.00752 (Gu, Dao)
- Mamba-2 / SSD: Transformers are SSMs (state space duality) — arxiv:2405.21060 (Dao, Gu)
- GLA: Gated Linear Attention — arxiv:2312.06635 (Yang et al.) — data-dep diagonal gate, chunkwise matmul-friendly
- Hyena: Towards Convolutional Language Models — arxiv:2302.10866 (Poli et al.) — implicit long conv via small FFN, FFT-conv
- RetNet — arxiv:2307.08621 (Sun et al.) — special case of SSD with constant scalar decay
- GateLoop — arxiv:2311.01927 (Katsch) — concurrent with SSD

### Mamba-3 (bleeding-edge, paper not stable)
- Mamba-3 Part 1 (Tri Dao, 2026): https://tridao.me/blog/2026/mamba3-part1/
- Mamba-3 Part 2: https://tridao.me/blog/2026/mamba3-part2/
- Goomba Lab Mamba-3 cross-post: https://goombalab.github.io/blog/2026/mamba3-part1/

### Hybrids
- Jamba: A Hybrid Transformer-Mamba — arxiv:2403.19887
- Zamba: Compact Mamba-Based Hybrid — arxiv:2405.16712
- Samba — arxiv:2406.07522
- Hymba: Hybrid-head Architecture — arxiv:2411.13676

### Quantization
- Quamba: Post-Training Quantization for SSMs — arxiv:2410.13229
- Quamba2 W4A8 — arxiv:2503.22879

### Limitations / weaknesses
- Zoology: Measuring and Improving Recall in Efficient Language Models — arxiv:2312.04927
- The Illusion of State in State-Space Models — arxiv:2404.08819

### Surveys
- A Survey on State Space Models — arxiv:2404.09516

```

(Then a blank line, then the `## Transformer/optimizer techniques retained (for hybrid composition)` header, then the existing PAPERS.md content verbatim.)

### 3.9 `TECHNIQUES_INDEX.md` (additive edit at root)

**Read the current file first** to see the existing records summary content. Prepend (above all existing content) this SSM section, and wrap the existing records-table content under `## Records-validated transformer techniques (for hybrid composition with SSM blocks)`.

New content to prepend:

```markdown
# Techniques index

## SSM technique families (current focus)

One-liner each; drill via PAPERS.md or references/INDEX.md.

- **S4 / S4D / DSS** — diagonal state-space, FFT-conv parallel training, RNN inference. Two-line init from HiPPO; LTI (no input-dependence). Debug-friendliest starting point.
- **Mamba-1** — selective scan with input-dependent (A̅, B̅, C, Δ). Sequential on MPS; vendored at `references/mamba_minimal_model.py`. Sharp LR cliffs documented (primer §4.2).
- **Mamba-2 / SSD** — scalar-per-time A_t, allows much larger N (state dim 64-256). Matmul-only, MPS-compatible. Less expressive per state but parallel-friendly.
- **GLA** — data-dependent diagonal gate, chunkwise matmul-friendly. Concurrent with SSD.
- **Hyena** — implicit long convolution via small FFN, FFT-conv. Different family but same "linear sequence modeling" goal.
- **RetNet / GateLoop** — special cases / concurrent variants in the SSD family.
- **Hybrid (Jamba/Zamba/Samba/Hymba)** — mix attention + SSM. On parameter golf leaderboard, Hymba (1.1828) and S4D-Lin hybrid (1.1682) both lost to contemporaneous transformers by 0.06–0.10 BPB. Cautious-known-losing has low upside; consider less-tried families too.
- **BigramHash** — adds effective vocabulary at near-zero cap cost. Record-validated. Per primer §4.5, candidate remedy for SSM's associative recall gap (Zoology shows 82% of the gap is recall).

```

(Then a blank line, then the `## Records-validated transformer techniques (for hybrid composition with SSM blocks)` header, then the existing TECHNIQUES_INDEX.md content verbatim.)

### 3.10 `.claude/skills/derive-and-verify/SKILL.md` (new — full content)

```markdown
---
name: derive-and-verify
description: Invoke when about to derive an SSM equation, kernel formula, discretization, or non-default initialization in scratch/ — before writing the train_gpt.py implementation. Carries the patterns for math-heavy research code (worked tiny example, cite reference formula, recurrence-vs-convolution as free oracle, init invariants, degenerate cases, print spectra) so silent bugs surface upstream rather than as a step-50 NaN. Especially apt for SSMs where representations mix and the recurrence amplifies errors. Distinct from pull-out (which is mode-shift, not how-to-math).
---

# Derive and Verify

The SSM literature mixes representations (continuous vs discrete, real vs complex, recurrence vs convolution) and the recurrence amplifies math errors over the sequence length. Silent bugs are common. This is the discipline you bring to `scratch/` before writing the implementation — patterns to adapt, not a procedure to follow.

**The cheapest debugging is the kind you do before training.** The recurrence will surface mistakes downstream as a NaN at step 50 or a worse-than-baseline val_bpb — but those mistakes were already visible upstream; you just hadn't looked.

## When to invoke
- Before deriving any new SSM equation in `scratch/` (discretization, kernel formula, selective scan, etc.)
- Before writing a custom selective-scan, kernel constructor, or non-default initialization
- Before initializing parameters with a specific formula (HiPPO-LegS, A_log, dt_bias) where getting the formula wrong is a silent failure
- After reading a primer section that introduces a new equation you'll implement

## When NOT to invoke
- Routine env-var tweaks
- Reading existing code (use `search_journal` or just `Read`)
- Trajectory-pattern checks during a run (`launch-and-await` covers that)
- Re-using a previously-verified block in a new experiment

## The patterns

### 1. Worked tiny example first
B=1, L=4, D=2, N=2. Compute by hand on paper. Compare to your code element by element. If you can't compute it by hand, you don't understand the operation well enough to debug it later. This is not optional — it is the difference between deriving and guessing.

### 2. Cite the reference formula in code
Every key equation gets a comment with the source:
```python
# ZOH per primer §1.2: Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I) Δ B
```
The cite is the version-controlled source of truth; your code is the implementation. Future-you (or a subagent) verifies code against intent via the cite.

### 3. Free oracle: recurrence-vs-convolution duality
For LTI blocks (S4, S4D), given the same `(A, B, C, Δ)`: the recurrent rollout `x_k = Ā x_{k-1} + B̄ u_k` and the convolution `y = K * u` must agree on the same input. If they don't, one of your implementations is wrong. This is the cheapest sanity check available for any S4-family block you write.

For selective (Mamba-family) scans, use `references/selective_scan_ref.py` as the oracle — see `references/INDEX.md` for the protocol.

### 4. Print invariants on init (once, at construction)
- **Eigenvalues of Ā in the closed unit disk**: `torch.linalg.eigvals(A_bar).abs().max()` should be ≤ 1. The standard `A = -exp(A_log)` parameterization gives this for free post-discretization, but verify.
- **dt strictly positive**: if `dt = softplus(dt_proj(x) + dt_bias)`, print `dt.min()` to confirm > 0. A negative dt silently breaks the recurrence's stability properties.
- **Kernel decay** (for LTI): `K[-1].abs().mean()` should be small if your timescales are right. If `K[-1] ≈ K[0]`, your timescales are too long for the sequence length.

### 5. Closed-form degenerate cases
- A=0 → pass-through: y = D*u (or zero if D=0)
- B=0 → output stays at zero regardless of input
- Δ=0 → no state updates, output collapses to D*u
- C=0 → output is zero regardless of state

Set the parameter, run a forward pass, confirm the output matches the closed form. Quick "is the wiring right" checks before training.

### 6. Print spectra at first forward (then remove)
During the derive-and-implement phase only — not in production training loops. Print:
- Eigenvalue spectrum of Ā (check stability, check distribution)
- dt distribution (`dt.histogram(...)` or just min/max/mean)
- A̅ row norms (catches outlier states)
- Kernel decay shape for LTI (early indication of timescale match)

These tell you whether the math you derived is the math the code is computing. Remove the prints once you've confirmed agreement; you don't want them firing every forward in a 200-step run.

## After
With patterns satisfied, the implementation is ready for the experiment loop. Trajectory still has to look right (step 1 ≈ ln(vocab), monotonic descent — see `launch-and-await`), but a clean step-1-to-10 trajectory plus pre-training derivations together is much stronger evidence of correctness than either alone. For SSMs especially, see also program.md "SSM-specific harness facts" → late-NaN gate (step-100 await is non-optional).

## If a derivation fails
If the duality doesn't agree, an eigenvalue is outside the unit disk, dt prints negative, or the worked tiny example mismatches: **do not run the experiment.** Fix the math first. Update `scratch/` with what you found wrong. The cost of fixing math in `scratch/` is minutes; the cost of fixing it via training-time debugging is hours.

## Distinct from pull-out
| Skill | Purpose | When |
|---|---|---|
| `pull-out` | Mode-shift to higher-level reassessment | Every reflective transition |
| `derive-and-verify` (this skill) | Patterns for *how* to do math well in `scratch/` | When about to derive an equation |

Pull-out tells you to use scratch/ ("compute the parameter count, sketch the math"). This skill tells you *how* to do that well for SSM-flavored math.
```

---

## 4. Execution order

### Pre-flight (one cheap verify)
- [x] Verify mamba-minimal model.py is fetchable: `curl -sI https://raw.githubusercontent.com/johnma2006/mamba-minimal/master/model.py` (should return 200) — **200 OK**. Latest commit touching `model.py`: `03de542a36d873f6e6c4057ad687278cc6ae944d`.
- [x] Identify selective_scan_ref source path in state-spaces/mamba: `mamba_ssm/ops/selective_scan_interface.py` returns 200. Latest commit: `74729d0f6d9c2096407eda5562393ab70960a3f6`.
- [x] Check upstream state-spaces/mamba for a NOTICE file at the repo root: **404 — no NOTICE file present**, so no Apache-2.0 §4(d) preservation obligation. LICENSE present (Apache-2.0).

### Phase A — directory restructure
- [x] `mkdir -p references/`
- [x] `mkdir -p summaries/_archive_transformer/ journals/_archive_transformer/ walks/_archive_transformer/`
- [x] `mv "State-space models for parameter golf a first-principles primer.md" SSM_PRIMER.md`
- [x] `git mv summaries/2026-04-25_*.md summaries/_archive_transformer/` (3 files)
- [x] `git mv journals/2026-04-25_*.md journals/_archive_transformer/` (3 files)
- [x] `git mv walks/2026-04-25_*.md walks/_archive_transformer/` (2 files)
- [x] Deleted scratch transformer-axis files (4 files: brainstorm_big_directions, depth_recurrence_design, next_experiments, parking_lot). Note for future agent: `rm` is aliased to `rm -i` in this shell — used `/bin/rm -f` to bypass.
- [x] Appended `references/_external/` to `.gitignore`. Confirmed via `grep`.
- [x] Verified primer rename: `ls SSM_PRIMER.md` returns the file.
- [x] Verified primer section structure: Layer 1-5 + conclusion all present.

### Phase B — vendor references
- [ ] Fetch johnma2006/mamba-minimal `model.py` → `references/mamba_minimal_model.py`. Add header per §3.4.
- [ ] Fetch state-spaces/mamba `selective_scan_ref` → `references/selective_scan_ref.py`. Add header per §3.4. **Preserve any NOTICE content if upstream has one.**
- [ ] Verify both parse: `python3 -c "import ast; ast.parse(open('references/mamba_minimal_model.py').read())"` and same for `selective_scan_ref.py`.
- [ ] Verify `selective_scan_ref.py` is standalone — `python3 -c "import ast,sys; tree=ast.parse(open('references/selective_scan_ref.py').read()); print([n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)])"` should show only `torch` / no `mamba_ssm` internal deps. If it has internal deps, simplify before committing.
- [ ] **MPS smoke test for `mamba_minimal_model.py`** — catches CUDA-default assumptions that would silently fall back to CPU. Run from the worktree root:
  ```python
  python3 -c "
  import sys; sys.path.insert(0, 'references')
  from mamba_minimal_model import Mamba, ModelArgs  # adjust class name if upstream differs
  import torch
  args = ModelArgs(d_model=64, n_layer=2, vocab_size=100)  # adjust ctor signature if upstream differs
  m = Mamba(args).to('mps')
  out = m(torch.randint(0, 100, (1, 16), device='mps'))
  print('out shape:', out.shape, 'device:', out.device)
  assert str(out.device).startswith('mps'), 'fell back to CPU'
  "
  ```
  Adjust the import / constructor names to match upstream's actual API (the class may be named `MambaLM` or similar; check the file). If it raises a CUDA-only error, edit the vendored copy to remove the `.cuda()` calls or device-default assumptions before integration.
- [ ] Write `references/INDEX.md` per §3.3.

### Phase C — add new skills
- [ ] `mkdir -p .claude/skills/noise-floor-sentinel/`
- [ ] Write `.claude/skills/noise-floor-sentinel/SKILL.md` per §3.5.
- [ ] `mkdir -p .claude/skills/derive-and-verify/`
- [ ] Write `.claude/skills/derive-and-verify/SKILL.md` per §3.10.

### Phase D — content edits in place
- [ ] Edit `program.md` per §3.1: replace title block + "From Previous Session" block; replace "Reference baseline" section; insert "SSM-specific harness facts" section; edit "Setup (every session)" steps; replace "Reference materials" section. Preserve all other sections verbatim.
- [ ] Reset `journal.md` Current threads + Open questions per §3.2. Clear Stack of confirmed wins and Dead axes content (keep their headings, leave empty placeholder content). Leave the existing `## Entries (newest first)` marker (current line 56) untouched.
- [ ] Prepend SSM section to `PAPERS.md` per §3.8; wrap existing content under `## Transformer/optimizer techniques retained (for hybrid composition)`.
- [ ] Prepend SSM section to `TECHNIQUES_INDEX.md` per §3.9; wrap existing records content under `## Records-validated transformer techniques (for hybrid composition with SSM blocks)`.
- [ ] Light edit `.claude/skills/search_journal/SKILL.md` per §3.7.
- [ ] Light edit `.claude/skills/promote/SKILL.md` per §3.6 (one-sentence caveat in "Decide if it's actually a win" section).

### Phase E — verification
- [ ] grep for stale paths: `grep -rn "State-space models for parameter golf" --include="*.md" .` should return zero matches (the long-named primer was renamed).
- [ ] grep for `docs/` references: `grep -rn "docs/" --include="*.md" .` should return zero matches in tracked content.
- [ ] `grep -E "^##" SSM_PRIMER.md` — sanity-check primer section structure intact.
- [ ] `git status` — confirm move/add/delete pattern matches §2.
- [ ] `wc -l program.md` — target ~290 lines (was ~225; growth from "What you are NOT doing" + "SSM-specific harness facts" + Setup-step expansion + Reference-materials rewrite + Reference-baseline expansion). Don't panic at 280-310; do reconsider if <260 (sections may have been silently lost) or >340 (verbatim sections may have been duplicated).
- [ ] **Verify all top-level sections present in program.md**: `grep "^## " program.md`. Expected sections (in order): "What you are NOT doing", "Reference baseline", "SSM-specific harness facts", "Setup (every session)", "Time budget", "Permissions", "The experiment loop", "Auto-promote", "Hypothesis discipline", "Logging formats", "Soft constraints", "Regression sentinel", "Running experiments", "Subagent for code edits", "Wrapping a session", "Reference materials", "NEVER STOP", "When the human returns and explicitly asks you to STOP". If any are missing, the sectional Edit calls displaced a verbatim-preserved section — recover from `git diff program.md` before continuing.
- [ ] All new artifacts present: `ls references/INDEX.md references/mamba_minimal_model.py references/selective_scan_ref.py .claude/skills/noise-floor-sentinel/SKILL.md .claude/skills/derive-and-verify/SKILL.md`.
- [ ] All archives populated: `ls summaries/_archive_transformer/ journals/_archive_transformer/ walks/_archive_transformer/` (3, 3, 2 files respectively).

### Phase F — wrap
- [ ] Final review of this plan doc; confirm all checkboxes in Phases A–E are ticked.
- [ ] `rm SSM_SETUP_PLAN.md` (this doc, at repo root).
- [ ] Summarize what was done back to the user. **Note from executing agent:** the user (Tony) asked for commits at each step of this execution and confirmed moving the plan doc to the repo root so it's git-trackable through the run. Honor those instructions over the original plan's "do NOT commit unless asked" wording.

---

## 5. Resumption notes (in case context is wiped mid-execution)

If you are a new instance picking this up:

1. Read this file (`scratch/SSM_SETUP_PLAN.md`) in full first.
2. `git status` — what's already moved/added/deleted?
3. Check which files exist vs. expected end-state from §2:
   - `SSM_PRIMER.md` at root (renamed from "State-space models...md")
   - `references/INDEX.md`, `references/mamba_minimal_model.py`, `references/selective_scan_ref.py`
   - `.claude/skills/noise-floor-sentinel/SKILL.md`
   - `.claude/skills/derive-and-verify/SKILL.md`
   - `summaries/_archive_transformer/`, `journals/_archive_transformer/`, `walks/_archive_transformer/` populated per §2
4. Check if program.md and journal.md have been edited:
   - program.md should have a new `## What you are NOT doing` section and a `## SSM-specific harness facts` section
   - journal.md Current threads should mention "SSM-family noise floor: UNCHARACTERIZED"
5. Resume from the first unchecked checkbox in §4.

User's confirmed decisions are in §1; do not re-ask. Architectural plan with content is in §3; treat as authoritative.

---

## 6. Where each piece of content lives (placement map)

The previous design pass identified content gaps in the primer (G1–G11). After review, each gap has a single home in existing structure — **no new docs**. Tag = FACT (verifiable in code/data) or OPINION (interpretation; one researcher's view).

| Gap | Tag | Lives in |
|---|---|---|
| G1 — Throughput tax framing (PR #831 calibration) | FACT + OPINION | program.md "SSM-specific harness facts" |
| G2 — Primer is internally inconsistent | OPINION | journal.md Current threads (one bullet) + program.md "Reference materials" (one phrase) |
| G3 — Suggested first-week recipe | OPINION | journal.md Open questions (the 9 ranked items) |
| G4 — Scale deception (PR #1227) | OPINION + FACT | journal.md Open questions ("scale deception" sub-section) |
| G5 — Depth recurrence transfer | OPINION | journal.md Open questions ("depth recurrence transfer" sub-section) |
| G6 — `CONTROL_TENSOR_NAME_PATTERNS` triple-duty (fp32 train + fp32 quant + AdamW route) | FACT | program.md "SSM-specific harness facts" |
| G7 — `ssd_minimal.py` is matmul-only | FACT | references/INDEX.md (curl-on-demand) |
| G8 — `mamba.py` parallel `pscan` | FACT | references/INDEX.md (curl-on-demand) |
| G9 — Step-0 regression sentinel | practice | program.md "Setup (every session)" step 7 |
| G10 — Mamba-3 blogs | FACT | PAPERS.md (Mamba-3 sub-section) |
| G11 — Alternative families (GLA/Hyena/RetNet/GateLoop) | FACT | PAPERS.md (Selective and gated sub-section) |

**Style for OPINION items**: cite the source (primer §X, Issue #Y, PR #Z); use suggestion verbs ("the primer suggests", "consider"), not imperatives; tell the agent how to log disagreement (`Empirical update to ...: ...`); distinguish principle from H100-specific constant where both exist.

---

## 7. Final pre-execution sanity

- [ ] User has reviewed §1 decisions table and §3 content sketches
- [ ] User says "go" or equivalent
- [ ] Then: Phase A.
