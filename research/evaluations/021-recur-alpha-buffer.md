# Evaluation 021 — Recur-α as register_buffer, full pipeline

> **⚠ PROVISIONAL — results below are confounded by an α-value bug.**
> The `cb5cd78` commit hardcoded pass-3 L4 α = **0.96484375** in the buffer
> tensor, but 017's actual trained endpoint at that site is
> **0.97265625** (one 1/128 tick higher; 019 and 019b use the correct value).
> The training-loss divergence from 019b begins at loop activation, exactly
> when α starts being applied — consistent with the bug being a real
> contributor, not just pod variance.
>
> A corrected rerun is in flight on commit `dc0b5f8` (single-digit fix,
> same branch, same pod, warm inductor cache). The §"scientific finding"
> below (buffer-α costs TTT flexibility) cannot be defended until we see
> whether the fixed run still regresses. Treat this eval as describing
> the bugged run only; final decision should be made from the `dc0b5f8`
> artifact. See `runs/021-recur-alpha-buffer/seed_42_fix/` when available.

**Spec:** `research/specs/021-recur-alpha-buffer.md`
**Runs:**
- `runs/021-recur-alpha-buffer/seed_42_8h_jp/` — **8×H100 JP authoritative** (this evaluation, **buggy α**)
- `runs/021-recur-alpha-buffer/seed_42/` — 4×H NE-1 partial (earlier run, kept for reference, also buggy α)
- `runs/021-recur-alpha-buffer/seed_42_fix/` — **IN FLIGHT** on `dc0b5f8` with corrected α
**Date:** 2026-04-21 (results) / 2026-04-22 (confound flag added)
**Commit:** `cb5cd78` (α as `register_buffer` frozen at 017 endpoint values — **pass-3 L4 off by 1/128**)
**Status:** full pipeline completed — train + pre-quant + GPTQ + TTT + serialize all landed, but superseded pending fix-run

---

## Result summary

| metric | value |
|---|---|
| Hardware | 8×H100 SXM AP-JP-1 ($23.92/hr, ~30min = ~$12) |
| Final step | 4883 / 20000 (wallclock cap 596s) |
| Loop activation | step 2156 @ frac 0.350 |
| val_bpb @ step 4000 | 1.1136 |
| val_bpb @ step 4883 (in-training) | 1.0701 |
| Pre-quant post-EMA val_bpb | **1.06963** |
| Post-GPTQ (pre-TTT) val_bpb | **1.07913** |
| **Post-TTT val_bpb** | **1.06900** |
| Submission size | 15.95 MB |

---

## Comparison to prior runs

| run | hardware | final step | pre-quant post-EMA | post-TTT val_bpb |
|---|---|---|---|---|
| #1736 reference | 8×H JP | — | — | **1.06610** |
| 019b | 8×H JP | 4824 | 1.06951 | **1.06628** |
| 017 | 8×H JP | 4697 | **1.06733** | 1.06733 |
| 019 | 8×H JP | 4697 | 1.07063 | 1.06744 |
| **021** | **8×H JP** | **4883** | **1.06963** | **1.06900** |
| 008 #1736 reproduction | 8×H JP | 4828 | 1.06922 | — (TTT not run) |

**Δ on headline metric (post-TTT):**
- 021 vs #1736 reference: **+0.00290** (021 worse — does NOT beat #1736)
- 021 vs 019b: +0.00272 (021 worse)
- 021 vs 019: +0.00156 (021 worse)
- 021 vs 017: +0.00167 (021 worse)

**021 is the slowest post-TTT of all 8×H runs in the recur-α family.**

---

## The scientific finding (PROVISIONAL — confounded by α bug): buffer-α trades TTT headroom for pre-TTT quality

Pre-TTT, 021 was actually *ahead* of 019 on val_bpb (1.06963 vs 1.07063). The headline regression appears entirely at the TTT step. Side-by-side TTT deltas:

| run | post-quant (pre-TTT) | post-TTT | **TTT Δ** |
|---|---|---|---|
| 019 (literal-α) | 1.07989 | 1.06744 | **−0.01245** |
| **021 (buffer-α)** | 1.07913 | 1.06900 | **−0.01013** |

**021's TTT recovers 0.0023 less val_bpb than 019's TTT.** Mechanistic read: frozen α (in a register_buffer) means TTT's LoRA adapters cannot modulate the α blending values during per-document adaptation. They have to route all corrections through projection weights. 019's literal-α gets the same frozen treatment structurally, but the graph doesn't tell Dynamo about the value (it's a literal, not a buffer), so torch.compile may inline it and leave more rewrite flexibility — OR the endpoint α values themselves are just in a better basin when treated as literals. Either way, TTT adapts better in 019 than in 021.

This is the real scientific content of spec 021. **Buffer-α is cleaner for throughput (0 Type B spikes, confirmed again here and in 020b) but is a mild val_bpb regression at the full-pipeline endpoint.**

---

## Throughput: buffer-α stability confirmed in production

Same clean profile we saw on 020b and 021-4H-NE-1:
- 0 Type B mystery spikes throughout the run
- Pre-loop steady at 8.13M tok/s (faster than 008/017/019 by 0.01-0.08M)
- Post-loop cumulative tok/s tracks 019 closely, but without 019's Type B chop
- 55 more steps than 008 (4883 vs 4828), 186 more than 019 (vs 4697)

Throughput is the win. Endpoint val_bpb is not.

---

## Step-matched loss profile

Post-loop (021 loop@0.35 vs 017/019 loop@0.45, so 021 activates 1300 steps earlier) — 021 runs **+0.005 to +0.020 higher loss** at matched steps. This is consistent with less pre-loop convergence before loop begins:

| step | 019 | 021 | Δ |
|---|---|---|---|
| 2500 | 2.5520 | 2.5617 | +0.010 |
| 3000 | 2.5644 | 2.5759 | +0.012 |
| 3500 | 2.5640 | 2.5724 | +0.008 |
| 4000 | 2.3981 | 2.4165 | +0.018 |
| 4500 | 2.2626 | 2.2857 | +0.023 |

Yet post-EMA at stop, 021 is *ahead* because the 186 extra steps compensated the per-step gap. TTT then reverses the order.

---

## Decision

**PROVISIONAL — pending the `dc0b5f8` fix-run. Re-evaluate on its post-TTT number:**
- If fix-run post-TTT ≤ 1.06610 → the bug was the cause; promote buffer-α, launch seeds 43/44.
- If fix-run post-TTT in (1.06610, 1.06710] → borderline; hold.
- If fix-run post-TTT > 1.06710 → the current decision stands; shelve buffer-α.

**Current recommendation (for the buggy run only):** Do NOT promote buffer-α as a val_bpb improvement. Keep it as a diagnostic tool (proven throughput stability, useful when chasing Type B spikes in other specs) but do not use it as the α container in submission runs. 019b (literal-α manual lerp) remains our closest miss to #1736.

### What to try next
1. **Literal-α + loop@0.35** — decouple the two variables. Is the early-loop-activation helping or hurting on 019b-style literal-α? If this beats 019b's 1.06628, early loop is the real lever.
2. **019b 3-seed rerun** — the 0.00018 miss to #1736 is within seed-std (~0.0002). A second seed ties or wins.
3. **Extra-depth TTT (spec 022)** — still un-blocked once the OOM is tuned for 8×H (revisit batch size)

### What NOT to try again
- Buffer-α on top of any loop-timing variant. The mechanistic story (buffer removes TTT flexibility) likely holds across loop schedules.

---

## Bugs / environment

- **brotli installed successfully** via `pip install --break-system-packages brotli` in preflight — NO post-training crash (fixed relative to 020b and 021-4H-NE-1)
- `pyminify unavailable` warning at Code-size measurement — non-blocking, just skips the compressed-code-size metric
- Spec's pinned commit `cb5cd78` pulled cleanly after git stash (pod had residual uncommitted changes from a prior session's spec 019 work)

---

## Cost accounting

- 8×H pod runtime: ~30 min × $23.92/hr ≈ **$12**
- Plus earlier 4×H NE-1 partial run: ~$3
- Plus probe pod overhead throughout the day: ~$30 (see memory: probe-cleanup leak)
- Spec 021 total: ~$15, plus $30 wasted on probe leak.

---

## Artifacts

All in `runs/021-recur-alpha-buffer/seed_42_8h_jp/`:
- `train.log` (41KB) — full training + TTT trace
- `diag_nvsmi.csv` (1MB)
- `final_model.int6.ptz` (16MB) — the submittable quantized model
- `final.json` — metrics
- `notes.md` — per-run notes
- `37a90c58-*.txt` — torchrun trace

`final_model.pt` (130MB) intentionally not rsynced.
