# 2026-04-22 — Frontier Scan (incremental)

Incremental scan from 2026-04-21T15:00:00Z. Five new PRs; three non-record (skipped), two record-track with completed results, one record-track with results pending.

## What's new

**PR #1767 (renqianluo) — Alpha=144 LoRA + Warm-start A + WD 1.0 — 1.07209 (3-seed)**

renqianluo submitted a clean, well-ablated set of four changes to `BatchedLinearLoRA` in the phased TTT:

1. Alpha/rank output scaling (`forward(x) * alpha/rank`) — standard LoRA trick, enables stable rank increase
2. Warm-start A across batches — A accumulates feature directions over ~780 TTT batches; B still zeros each batch, so per-document LoRA output is zeroed (legal)
3. WD 1.0 counteracts A overfit (doubling from 0.5)
4. Alpha lifted 96→144 with rank 128 (effective scale 1.125)

3-seed mean: **1.07209**, every seed monotonically improving. Their baseline stack is non-CaseOps (~1.073), not #1736 (1.065). So the absolute number isn't actionable, but the warm-start-A concept is novel and composable.

**This is a genuine, extractable eval-time lever we don't have specced.** Adding to ideas as `ttt-lora-warmstart-a`.

**PR #1765 (renqianluo) — intermediate step (1.07266)** — superseded by #1767, same concept without alpha=144.

**PR #1766 (tashapais) — Recur-Alpha on #1736 stack — pending results**

tashapais independently implemented our spec 021 Recur-Alpha concept (buffer carry scalar, init=0, `x += α * carry_first`). Code is identical to our spec 021 mechanism. Results pending 8×H100 run. If she reports below 1.065, that's independent corroboration. Worth watching.

Notable: she co-authored with Claude Sonnet 4.6 (per commit line), running a similar research loop.

## Leaderboard unchanged

Best clean: **#1756 @ 1.06505** (romeerp, depth curriculum). Our baseline: **#1736 @ 1.06549**. No new clean PR below 1.065.

## Priority for today

1. Spec 021 (recur-alpha-buffer) is READY — main priority is pod execution
2. New idea: `ttt-lora-warmstart-a` (LoRA warm-start A) — worth a quick spec after 021 results land
3. Watch #1766 for results — independent signal on spec 021 hypothesis
