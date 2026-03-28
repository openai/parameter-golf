# Parameter Golf — official time limits (leaderboard / record track)

From the [project README](../README.md) (FAQ):

- **Training:** leaderboard submissions must finish training in **≤ ~10 minutes on 8× H100 (SXM)**.
- **Evaluation:** submissions must **also** complete evaluation in **≤ ~10 minutes on 8× H100**. This is **in addition to** the training-time limit — not a single 10-minute block for both.

So for a **record submission**, both phases must respect their own caps on the stated hardware.

**Local research** (e.g. 1× RTX 3090, longer TTT eval wall time) does not change the official rule; before submitting, ensure your **training + eval** pipeline fits the challenge harness and stays within both limits on **8× H100**.
