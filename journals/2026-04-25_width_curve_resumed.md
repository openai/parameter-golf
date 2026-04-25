# Journal entries — 2026-04-25 (late afternoon, width-curve resumed session)

Rotated from `journal.md`. See `summaries/2026-04-25_width_curve_resumed.md` for the narrative handoff. Earlier same-day entries are in `journals/2026-04-25_code_directions.md`.

## 2026-04-25 · exp 0063 · K=3 L=3 + SwiGLU(mlp=11) wins +0.007 over 0062 — NOT promoted (unconfirmed pile-up)

**Question**: Push the MLP-width curve to mlp=11 (cap edge). Does the monotonic gain continue or saturate?

**Setup**: Forked 0062. Single env-var: `MLP_MULT=11` (28.4M params raw, est ~16 MB int8). All else identical to 0062.

**Prediction** [CONJECTURE]: Δ -0.005 to +0.010 vs 0062. Diminishing returns expected.

**Result**: val_bpb_post = **2.07994**, Δ = **+0.00693 vs 0062** (judgment-call +0.005-0.010), Δ vs 0057 mean = +0.0243. Pre-quant 2.0773 — pre/post Δ match. Quant_tax 0.00263 (normal). Artifact **15.851 MB** (12 KB under the 16 MB cap — extremely tight). Step_avg 6368 ms.

**Methodology stop — NOT PROMOTED**: 0059 (mlp=4), 0062 (mlp=8 — committed promote), and now 0063 (mlp=11) are all single-seed direct-promote-zone wins. The 0057→0058 SEED=42 confirmation discipline was applied once and dropped. Cumulative single-seed gain since canonical: +0.441 (~17.5%). But SEED=42 confirmations are an unpaid debt. Per protocol guidance from external review, **before any further architectural push, the next experiment must be SEED=42 of 0062**. The 0063 row stays `keep` (above noise floor) but is NOT promoted to winners/.

**Conclusion** [CONJECTURE]: width keeps paying through mlp=11 at the cap edge. But three unconfirmed seeds in a row means the magnitude estimate is uncertain by potentially 0.01-0.02 (cf. last session's lesson that single-seed Δ at the boundary overstates by ~10-20%). The cumulative could be anywhere from +0.40 to +0.45 once cross-seed confirmed.

**[transfer:high]** for the trend; magnitude depends on confirms.

## 2026-04-25 · exp 0062 · K=3 L=3 + SwiGLU(mlp=8) — NEW BEST 2.08687, Δ+0.0102 vs 0059 mlp=4

**Question**: With 0059 (mlp=4) showing the wider-MLP curve still paying inside recurrence, does pushing further to mlp=8 (hidden=4096 per block) keep paying or hit a capacity ceiling?

**Setup**: Forked 0057. Single env-var change: `MLP_MULT=8`. SwiGLU MLP: w_gate(d, 8d) + w_up(d, 8d) + w_down(8d, d). 21.3M params raw, expected artifact ~13 MB.

**Prediction** [CONJECTURE]: Δ +0.005 to +0.020 vs 0057. Wider keeps paying based on 0059 result.

**Disconfirming**: Δ ≤ -0.005 vs 0057 → very wide MLP under-trains at 200 steps.

**Result**: val_bpb_post = **2.08687** — direct-promote-zone win.
- **Δ vs 0059 mlp=4 (current preliminary winner): +0.01019** (above +0.010 noise floor → direct-promote)
- Δ vs 0057 SEED=1337: +0.01588
- Δ vs 0057 mean: +0.01740 (huge against the mean-anchor)
- Δ vs 0051 (original winner): +0.02284
- Pre-quant Δ vs 0059: +0.0114 (matches post-quant; clean training gain)
- Quant_tax 0.00247 (normal). Artifact **12.24 MB** (vs cap 16 MB; 3.76 MB headroom).
- Step_avg 5396 ms (vs 0059's 4206 ms — 28% slower per step due to 2× wider MLP).

Trajectory comparison vs 0059 / 0057:
- Step 100: 0062 3.76 vs 0059 ~3.81 vs 0057 ~3.82
- Step 200 train_loss: 0062 3.50 vs 0059 3.52 vs 0057 3.55

**Conclusion** [LIKELY, pending SEED=42]: Width-inside-recurrence is the dominant lever. Each unit of mlp_mult adds ~+0.002 of val_bpb improvement. The curve is monotonic from mlp=3 (2.103) → mlp=4 (2.097) → mlp=8 (2.087). The compound (recurrence × wide gating) is much more powerful than either alone.

**Cumulative gain vs canonical (2.5212)**: **+0.434 (~17.2%)**. Best result of the day.

**[transfer:high]** — wider MLP is the most robust H100-transfer lever; this should hold cleanly at 20K-step training.

**Followups**:
1. SEED=42 confirm of 0062 (mandatory for direct-promote within 5 experiments).
2. Push curve: mlp=10 (~14.7 MB, fits) or mlp=11 (~16 MB, borderline). The marginal gain may diminish but still likely positive.
3. Combine: K=3 L=4 (more loops) + SwiGLU(mlp=8) → eff depth 12, but artifact stays at 12.24 MB.

## 2026-04-25 · exp 0059 · K=3 L=3 + SwiGLU(mlp=4) — NEW BEST 2.09706 (Δ+0.007 vs 0057 mean)

**Question**: Does SwiGLU MLP_MULT=4 (vs 0057's mlp=3) inside K=3 L=3 recurrence pay? 0057's cap is at 6 MB; 10 MB headroom is unused. mlp=4 → 7.3 MB artifact, still well under cap.

**Setup**: Forked 0057. Single env-var change: `MLP_MULT=4`. No code change. SwiGLU MLP architecture same (w_gate + w_up + w_down).

**Prediction** [LIKELY]: Δ +0.005 to +0.020 vs 0057 — wider per-block MLP inside recurrence should help, records cap at mlp=3-4 in their stacks.

**Disconfirming**: Δ ≤ -0.000 vs 0057 — wider hurts at recurrence.

**Result**: val_bpb_post = **2.09706** (vs 0057 SEED=1337 2.10275, mean 2.10427).
- Δ vs 0057 SEED=1337: **+0.00569** (judgment-call zone)
- **Δ vs 0057 mean: +0.00721** (above noise floor)
- Δ vs 0051 mean (orig benchmark): +0.0127 (clear win)
- Pre-quant Δ vs 0057 SEED=1337: +0.0046 (matches post-quant; clean training gain)
- Quant_tax 0.00126 (low-normal, similar to 0057's 0.0024). Artifact 7.28 MB.
- Step_avg 4206 ms (vs 0057's 3721 ms — 13% slower per step due to 33% wider MLP).

Trajectory comparison: at step 165, 0059 train_loss = 3.5397 vs 0057 3.5533 (0059 +0.014 ahead). Final step 200 train_loss 0059 = 3.52 vs 0057 = 3.53.

**Conclusion** [LIKELY, pending SEED=42]: Wider per-block SwiGLU pays inside recurrence. The mlp=3 → mlp=4 increase is a 33% expansion of hidden dim per block (1536 → 2048), giving the gating more interaction capacity. Records that cap at mlp=3-4 standalone don't have the recurrent invocation pattern; with K=3 L=3, each MLP runs 3× per token, amplifying the value of more capacity.

The headroom story matters: cumulative gain since canonical (2.5212) is now **+0.424 (~16.8%)**. Cap is at 7.28 MB — still 8.7 MB headroom for more compounds. The strategic position keeps improving.

**[transfer:high]** — wider MLP is well-known robust at H100 scale; this is exactly the kind of "mlp_mult=4 wins" finding that transfers cleanly.

**Followups**: 0062 (mlp=8, launching) tests if even wider helps; SEED=42 confirm of 0059 mandatory before final promote.

## 2026-04-25 · session resumed at ~15:55 EDT

User pointed out I prematurely closed the session at 15:17 after misreading "find time to wrap up what you have and go take-a-walk" as a session-end signal. Per program.md, that was a walk request, not a stop request. NEVER STOP until manually told. Resuming on the highest-EV next move from the walk note: **K=3 L=3 + SwiGLU(mlp=8)** as the cap-utilizing push.
