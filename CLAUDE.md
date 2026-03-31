# Parameter Golf Lab — Agent Protocol

## Orient first
```
cat neural/LEADER.md    # current neural SOTA
cat crawler/LEADER.md   # current crawler SOTA
```
These two files tell you where the lab stands. Read them before doing anything.

## Repo structure
```
neural/      ← Neural SOTA track (Rascal lineage) — leaderboard #1 focus
crawler/     ← Crawler track (Bandit_Wagon lineage) — compression/quality focus
vault/       ← Immutable locked sources. Never modify.
records/     ← Leaderboard submission records. Never modify.
scripts/     ← Shared runners. sota_now.sh is the neural baseline runner.
data/        ← Dataset. Never modify.
junkyard/    ← Legacy experiments. Read-only reference only.
```

## Hard rules

**NEVER overwrite a test file.** Always create a new file. If you need to modify
a training script, copy it first, work on the copy, name it clearly.

**Confirm names before creating.** Ask the user what to name a new leg, script,
or directory before creating it. Never invent names silently.

**ONE variable per test.** If a run changes more than one thing vs the baseline,
the result is uninterpretable and the money is gone.

**Gate before 8x.** Every hypothesis runs a 1-GPU 2000-step gate (~$0.50) before
an 8×H100 full run (~$3-4). Never skip the gate.

**Never submit from TEST_LAB.** Submissions go: dedicated branch → fork1 → PR.

## RunPod workflow
1. Pod always pulls from `TEST_LAB` branch
2. Commit and push scripts BEFORE launching the pod
3. On pod: `git pull && bash <script>`
4. Never push FROM the pod
5. Pod gets destroyed after the run — save checkpoints before destroying

## Test cycle: Hypothesis → Ablation → Results

Every leg follows this sequence. No skipping steps.

```
hypothesis.md    ← write FIRST. ONE variable. Why. Gate target.
train_gpt.py     ← copy from leader, make the ONE change
gate.sh          ← commit+push → pod pulls TEST_LAB → run (1-GPU, 2000 steps)
ablation.md      ← fill gate result. Pass? Proceed. Fail? Stop.
run.sh           ← commit+push → pod pulls TEST_LAB → run (8×H100, 600s, seed=444)
ablation.md      ← fill full run result. Beats leader? Run confirmation.
                    confirmation run (8×H100, 600s, seed=300)
RESULTS.md       ← verdict (PROMOTES / DOES NOT PROMOTE), what we learned, next hyp
```

New legs are scaffolded with all three files pre-created:
```bash
bash scripts/new_leg.sh neural <name>
bash scripts/new_leg.sh crawler <name>
```

## Seeds
- Primary: 444
- Confirmation: 300
- Never use 1337

## Cost
- 8×H100 SXM: ~$13.36/hr
- Full 10-min run: ~$3-4
- Gate (1-GPU, 2000 steps): ~$0.50
- Do not suggest a run without a validated gate or clear hypothesis
