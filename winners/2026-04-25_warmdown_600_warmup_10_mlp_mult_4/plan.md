# Experiment 0008_mlp_mult_4_on_winner

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_600_warmup_10`)

## Question
0007 with MLP_MULT=3 produced a marginal Δ=+0.006 vs the winning baseline
(judgment-call zone). MLP_MULT=4 doubles the per-block hidden dim (vs
canonical mlp_mult=2) — substantially more capacity. The records repeatedly
hit MLP_MULT=4 in winning configs (e.g. 2026-04-01 SP4096+MLPMult4 at 0.9979).

Does MLP_MULT=4 produce a clearer Δ that resolves the marginal-result
ambiguity in 0007? Either way (clear win or clear flat), it tells us whether
capacity scales monotonically or hits a ceiling at sp1024 / 9L / d=512.

## Hypothesis [LIKELY]
Δ vs winner ≈ +0.012 to +0.025. If 0007's +0.006 was real signal at half
this capacity bump, MLP_MULT=4 should land cleanly above the +0.010
noise floor. If capacity already saturated at MLP_MULT=3, expect Δ flat
or only marginally larger than 0007's +0.006.

Param/artifact estimate (relative to canonical baseline):
- Per-block MLP at mlp_mult=4: fc(512×2048) + proj(2048×512) = 2.1M params.
- Δ per block ≈ 1.05M (vs canonical mlp_mult=2); 9 blocks → +9.4M params.
- Compression ratio observed in 0007: 1.86 MB additional artifact for +4.7M
  params (vs winner). Extrapolating linearly: +9.4M params ≈ +3.7 MB beyond
  winner. Predicted artifact: 8.105 + 3.7 = ~11.8 MB. Comfortably under
  16 MB cap.

## Change
`env.sh`:
- `LR_WARMUP_STEPS=10`
- `WARMDOWN_ITERS=600`
- `MLP_MULT=4`

No code edits.

## Disconfirming
- Δ ≤ +0.005 (noise vs winner): capacity saturates between 2 and 4 — try
  ablating other axes (sequence length, depth, attention).
- Δ < 0007's +0.006: more capacity hurts on the smoke — possibly because
  bigger MLP has more under-trained params at 200 steps + needs even more
  warmup for stability. Would deprioritize further capacity.
- Δ ≥ +0.020 (clear win): great. SEED=42 re-run if Δ ≥ +0.050; otherwise
  promote.
- size_violation: math says 11.8 MB ≪ 16 MB; if hit, our compression
  estimate was off by 35%+, would warn for future capacity decisions.
- NaN: bigger MLP + same schedule pushes optimization off the stable
  manifold.

## Notes from execution
<!-- Filled after the run. -->
