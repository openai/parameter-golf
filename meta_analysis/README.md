# Meta-Analysis: Mining 975 Expensive Training Runs

A data analysis of all 409 submissions (975 training runs) to the 8×H100 track.

## The idea

8×H100 time is expensive (~$40 per 10-minute run). But 975 teams already burned that compute, and all the logs are public. Can we mine this data to predict how a submission will perform on 8×H100s without actually running 8×H100s?

Three ways this could work:

1. **From early checkpoints** — if the outcome is visible at step 1,000, you can kill bad runs early
2. **From source code** — if technique choices predict the final score, you can estimate BPB before training
3. **From cheaper hardware** — if a run on an A6000 or RTX 5090 correlates with 8×H100 results, you skip the expensive machine entirely

## Part 1: The Dataset (this PR)

Built a database from all 409 PRs. Classified every submission using a three-step pipeline:

1. A Claude agent examined PRs and built a taxonomy from scratch (8 strategies, 49 techniques, 5 violation types)
2. A script tags each PR against that taxonomy (one LLM call per PR, structured output)
3. A deterministic script merges tags + training logs + architecture metadata into the final database

Split: 264 valid, 132 invalid (93% of N-gram submissions broke the rules), 13 unauditable.

Key findings from the valid runs:
- The shared template turned 975 runs into a natural controlled experiment
- BigramHash, EMA, XSA had the strongest technique associations
- 25–30M parameters at int6 is the sweet spot under 16MB
- **Early BPB at step 1,000 correlates 0.86 with final BPB. Seed variance is 0.5 mBPB. You can see the outcome 90 seconds in.**

That last point is what makes Parts 2 and 3 possible.

## Part 2: The 1,000-Step Oracle (coming soon)

A model that predicts final BPB from early training checkpoints.

## Part 3: Cross-Hardware Prediction (coming soon)

Train on an A6000 or RTX 5090, predict what 8×H100s would produce.
