---
name: agent-lab
description: Parameter Golf agent-lab — researcher mindset, experiments registry, commits, metrics. Use for agent_lab/ work and agent-lab build logs.
---

# Agent lab (Parameter Golf)

## Researcher mindset (embody this)

Work like a scientist, not only a tuner.

- **Loop:** question → **hypothesis** → implement → run → **measure** → reflect → new question. Write the hypothesis in `experiments.tsv` and the commit body before you romanticize the result.
- **Scope:** early runs can be **hyperparameters / schedule / batching** to learn the stack. Then deliberately move to **architecture and training mechanics**: attention patterns, block design, MLP/activations, **why this optimizer**, alternatives from papers or your own guesses, **quantization / QAT / low-precision** paths — always within challenge rules and honest **`val_bpb`**.
- **Breadth + depth:** don’t only stack wins on one path. From a **shared parent** commit, try **sibling** one-factor experiments (breadth). Go deeper on promising lines. Add **combo** runs when you suspect **interaction** (two ideas that only work together).
- **Evidence:** negative results belong in the log. Confusion is data.

## Before changing code

1. Read **`agent_lab/program.md`** (hard constraints).
2. Read **`agent_lab/experiments.tsv`** — what was tried, verdicts, best commit so far.
3. Read **`.cursor/rules/parameter-golf.mdc`** (challenge guardrails).

## After each full training run

1. Append **`agent_lab/results.tsv`** (gitignored loop log) if you use it — columns per `program.md`.
2. Append **`agent_lab/experiments.tsv`** (tracked) with stable **`AL-YYYYMMDD-NNN`** id, parent commit, hypothesis, **verdict** (`correct` / `wrong` / `partial` / `n_a`), metric, `val_bpb`, notes.
3. Commit with **`feat(agent-lab):`** or **`docs(agent-lab):`** and **rich body** (see **Commit conventions** below).
4. Update **`docs/build-logs/<date>-agent-lab.md`** — journal entry in a **human voice** (see **Build log voice** below).

## One-at-a-time vs interaction effects (important)

**Default loop:** change **one** thing between commits when you can — attribution is clean and matches a disciplined ablation.

**Reality:** some ideas only work **together** (e.g. smaller batch + higher LR). Pure hill-climbing can **discard** a change that looks neutral alone but is needed for a later combo.

**Practices:**

- Keep the **spine** of verified wins; occasionally spawn a **combo** experiment (`Exp: …-combo`) that stacks 2–3 pending ideas and compare to the best single-change line.
- If a single change **hurts** or is flat, note **`wrong` or `partial`** but add a **follow-up** row if you suspect **interaction** — don’t treat “no immediate gain” as permanently dead without a designed retest.
- Log **interaction hypotheses** explicitly in `experiments.tsv` notes and the build log so future you (or an agent) can see what was never tried together.

## Stable experiment IDs

- Pattern: **`AL-YYYYMMDD-NNN`** (NNN = 001, 002, … per day).
- Same ID in: `experiments.tsv`, commit body `Exp:`, build log headings.

## Primary metric and the three “final” lines

The training script prints several finals. Typical meanings:

| Log line | Meaning |
|----------|--------|
| **`final_int8_zlib_roundtrip`** | Model weights are **quantized to int8**, **zlib-compressed**, **decompressed**, loaded back into the model, then **standard validation** `eval_val` runs. Tests **compression + roundtrip correctness** and reports **`val_bpb`**. |
| **`final_int8_zlib_roundtrip_exact`** | Same metric, **more decimal places** (debug / tie-break), not a different method. |
| **`final_int8_ttt_lora`** | After the roundtrip weights, the script runs **test-time training** with **LoRA adapters** on the validation procedure (challenge-relevant path). Often the **primary** score for comparisons in this repo. |

**Lower `val_bpb` is better** for all. Don’t mix zlib-only vs TTT lines in the same leaderboard unless you document the switch.

## Artifact size (16 MB cap)

Submission size = **UTF-8 bytes of training code** + **compressed model** (see main README). **`Total submission size int8+zlib:`** in the log is a good sanity check. If you are at **~9.9 MB** compressed payload + code, you still have **headroom** toward **16,000,000 bytes** — room for **wider layers**, **more parameters**, or **less aggressive compression**, as long as train/eval still meet official limits.

## Official challenge time limits (leaderboard / record)

From the project README FAQ:

- **Training:** **≤ ~10 minutes on 8× H100 (SXM)** for record submissions.
- **Evaluation:** **≤ ~10 minutes on 8× H100** as well — **in addition to** training, not one combined 10-minute window.

**Local dev** (e.g. 1× 3090, long TTT wall time) is **not** proof you meet the official eval cap. Before claiming a record, run the **full** train + eval pipeline on **8× H100** (or the official harness) and confirm **both** phases fit.

## Commit conventions (Conventional Commits)

**Subject:** `<type>(agent-lab): <imperative short description>` — types: `feat`, `fix`, `docs`, `chore`, `refactor`.

**Body (recommended):** grep-friendly block:

```
Exp: AL-YYYYMMDD-NNN
Parent: <7-char sha>
Hypothesis: <one sentence>
Metric: final_int8_ttt_lora (lower better)
Result: keep | discard (val_bpb …)
```

Older commits may use non-conventional subjects; **new** work should follow this.

## Build log voice (for `docs/build-logs/*-agent-lab.md`)

Write like a **lab notebook**, not a press release. It is useful to:

- Say what you **expected**, what **confused** you, and what still **doesn’t make sense**.
- Admit **dead ends** and **silly mistakes** — that saves the next session from repeating them.
- Mix **plain-language** explanations (for learning) with **exact numbers and commits**.

If the prose sounds too polished, add one paragraph of **raw** “what I actually think.”

## Run script

From repo root, prefer **`./scripts/agent_lab/run_exp.sh`** (sets defaults for `RUN_ID`, `DATA_PATH`, `TOKENIZER_PATH`, activates `.venv` if present). Redirect logs yourself, e.g. `> agent_lab/run.log 2>&1`.

## Adapt this skill

When you discover friction (slow TTT, unclear metric, bad defaults), **edit this SKILL.md** or **`agent_lab/program.md`** so the next session inherits the lesson.
