---
name: arjunautoresearch
description: Pulls all open PRs from the openai/parameter-golf repo, ranks techniques by expected impact, composes the best combination, runs iterative experiments, and packages a submission.
metadata:
  author: arjun-krishna1
  version: "1.0"
---

# AutoResearch

Systematic workflow for distilling and composing the best approaches from the openai/parameter-golf challenge

## Phase 1: Gather all candidate approaches

Fetch every PR's metadata and diff:

```bash
scripts/pull-pr-diffs.sh pr_diffs 60
```

This saves `pr_<number>_view.txt` and `pr_<number>_diff.txt` for each PR. Read through all diffs and extract:

- **What** the technique does
- **Score** achieved (and whether it was verified on target hardware)
- **Side effects** on artifact size, step time, memory

Sort into three tiers:

| Tier | Criteria |
|------|----------|
| **High** | Verified improvement on target hardware, clean mechanism |
| **Medium** | Promising idea, but no verified run or unclear interactions |
| **Low** | Negative result, breaks constraints, or too architecture-specific |

## Phase 2: Compose the best combination

Stack High-tier techniques in order of expected gain. The hard part is catching interactions — techniques that work alone can hurt each other.

**General patterns to watch for:**

- **Redundant defenses**: If technique A already solves a problem (e.g. quantization gap), technique B targeting the same problem adds overhead for no gain. Remove B.
- **Memory budget conflicts**: Stacking changes that each increase memory (larger batch, longer sequences, bigger eval windows) can OOM. Estimate total memory before running.
- **Time budget tradeoffs**: Any change that slows per-step time means fewer total steps under a wallclock cap. The per-step quality gain must outweigh the lost steps.
- **Size budget math**: If the artifact has a size cap, changes that grow the model (larger vocab, fp16 tensors) need to be offset by trimming elsewhere (smaller hidden dims, fewer layers).

Write down the composed config before running. Note which techniques you kept, which you dropped, and why.

## Phase 3: Run, watch, iterate

Run the experiment and monitor the log in real time. Things to look for:

- **Step timing**: Is it in the expected range? Much slower than expected may mean an unintended overhead got compiled in.
- **Loss trajectory**: Compare to reference runs at the same step counts. If you're tracking behind, something is wrong.
- **Eval completion**: If the run finishes training but the eval crashes (e.g. OOM), you need to reduce eval batch size and rerun — don't waste a full training run.
- **Artifact size**: Check the final compressed size against the cap before celebrating the score.

After the first run, review what worked and what didn't. Drop anything that hurt, adjust parameters, and run again. Two to three iterations usually converges.

## Phase 4: Establish statistical significance

Most challenges require multiple seeds to prove the result isn't a fluke. Three seeds is standard. Run the same config with different random seeds, then:

```python
from math import sqrt

scores = [seed1_score, seed2_score, seed3_score]
mean = sum(scores) / len(scores)
std = (sum((x - mean)**2 for x in scores) / (len(scores) - 1)) ** 0.5
threshold = current_best - required_margin

t = (threshold - mean) / (std / sqrt(len(scores)))
# For 3 seeds (df=2): need t > 6.965 for p < 0.01 one-tailed
```

## Phase 5: Package the submission

Follow whatever format the challenge requires. Common elements:

- **README**: what you did, config, run command, metrics, significance stats
- **submission.json**: author info, score, artifact size
- **Train logs**: one per seed
- **Training script**: must be standalone and runnable from the submission folder

## Gotchas

- **Run command paths**: If reviewers run your script from the repo root, the run command in your README needs the full path to your script, not just `train_gpt.py`.
- **Leaking root files into the PR**: If you modified shared files during development, revert them before opening the PR. The submission should only add your folder.
- **Eval OOM on longer sequences**: If you trained at a longer sequence length than the baseline, the eval may need a much smaller batch size. A 4x longer sequence can need an 8x smaller eval batch.
- **Intermediate validation burns time**: Under a wallclock cap, every mid-training eval pass is time you could have spent training. Disable periodic eval for final scored runs.
