# LR Sensitivity Experiment — MATRIX_LR=0.08, 1×H100

**GitHub:** AbhiShet108
**Track:** Non-record
**val_bpb:** 1.3479
**Artifact size:** 14,672,726 bytes

---

## Experiment history

This is not our first run. We have completed multiple training runs
across this project, though not all logs were preserved. Earlier runs
were lost due to two issues: a pod volume that was wiped on termination
(before we switched to a network volume), and a template misconfiguration
that required starting over. These mistakes cost real compute budget and
are documented here honestly as part of the learning record.

The runs we can confirm:

1. **baseline_sp1024** — unmodified baseline on 1×H100. val_bpb: 1.3485,
   size: 12.3MB. This established our reference point on a single GPU.

2. **lr_matrix_08** — MATRIX_LR increased from 0.04 (default) to 0.08 (×2).
   val_bpb: 1.3479, artifact size: 14,672,726 bytes. This is the run
   being submitted.

Additional earlier runs exist but logs were not recoverable due to
the storage issues described above.

---

## What we changed and why

The only change from the unmodified baseline in this submission is
`MATRIX_LR=0.08`, doubling the default matrix learning rate of 0.04.

Hypothesis at the time: a higher learning rate might allow the model
to converge faster within the 10-minute wall clock cap, potentially
improving bpb on a single H100 where training is severely time-limited.

Result: marginal improvement from 1.3485 to 1.3479 — essentially
within noise. The learning rate change did not produce a meaningful
difference at this scale.

---

## What we learned

**Hardware is the dominant constraint at this scale.** The baseline
was trained on 8×H100s simultaneously, processing roughly 8× more
tokens in the same 10 minutes. On a single H100, our run reached
step 1145 of 20000 before the wall clock cap. Hyperparameter tuning
has limited value when the model has barely begun training.

**Free smoke tests on Colab T4 are misleading for timing.** We ran
many iterations on Colab during development. What takes minutes on
an H100 takes hours on a T4. This gap between free-tier testing and
real hardware is the most important practical lesson from this project.

**Storage discipline matters.** We lost earlier run logs because we
did not save them to a persistent volume or Google Drive immediately
after each run. Switching to a network volume and committing logs to
git at every session end fixed this — but only after paying the cost
of losing earlier results.

---

## Why the bpb is above the official baseline

Our result of 1.3479 bpb is above the official baseline of 1.2244.
This is expected and honest. The official baseline uses 8×H100s.
We used 1×H100. More parallelism means more tokens processed in the
same wall clock time, which means lower bpb. This is not a flaw in
our approach — it is a direct consequence of compute constraints.

---

## What we would try with more compute

- **Muon optimizer** — well-represented in top submissions, drop-in
  replacement for AdamW, no size impact
- **Depth recurrence** — loop the same transformer block N times,
  more compute per parameter within the 16MB budget
- **Vocabulary size 2048** — increase from 1024, costs ~0.5MB,
  hypothesis: fewer tokens per document improves context quality

With more compute the goal would be to run these as controlled
single-variable experiments — changing one thing at a time and
measuring the effect — rather than guessing which change helped.