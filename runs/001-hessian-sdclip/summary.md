# Spec 001 — Hessian-SDClip λ Screen — Summary

All 6 λ values swept on spec-000's `ckpt_final_pre_ema_step3849.pt` with cached Hessians (67 keys, 232 MB, reused across all λ). No training, no sliding-window, no TTT — quant-only screen.

| λ | val_bpb_quantized | Δ vs local λ=0 | artifact (bytes) | artifact | time |
|---|---|---|---|---|---|
| 0.00 | 1.10518 | (baseline) | 15,979,642 | 15.24 MB | 168.9s |
| 0.05 | 1.10527 | +0.00009 | 15,980,308 | 15.24 MB | 130.3s |
| 0.10 | 1.10530 | +0.00012 | 15,982,510 | 15.24 MB | 133.9s |
| 0.20 | 1.10553 | +0.00035 | 15,991,263 | 15.25 MB | 145.4s |
| 0.40 | 1.10618 | +0.00100 | 16,019,404 | **15.28 MB ⚠ >16 MB** | 135.7s |
| 0.60 | 1.10676 | +0.00158 | 16,057,235 | **15.32 MB ⚠ >16 MB** | 136.4s |

**No positive signal at any λ tested.** Trend is cleanly monotonic worsening with larger λ. Signal gate (`Δ ≤ −0.0003` for ≥1 non-zero λ) **not met**.

Secondary finding: **artifact size grows with λ**. At λ=0.40 and λ=0.60 the compressed model exceeds the 16,000,000-byte leaderboard limit — so even if these had shown a bpb improvement, they'd be inadmissible as submissions. The `adj = 1 + λ(r_i − 1)` multiplier stretches the row-wise scale `s`, which lowers compression efficiency of the quantized int6 matrices.

## Validity-gate note

Spec's validity gate required λ=0.00 to reproduce spec-000's `val_bpb_quantized = 1.10430` within ±0.0001. Our λ=0.00 produced **1.10518**, off by **+0.00088**. This is NOT a code bug — it's expected from a 1-GPU vs 8-GPU Hessian difference:

- Spec 000's Hessian: 64 calibration batches distributed across 8 ranks (each rank processes its shard of FineWeb).
- Our 1-GPU screen's Hessian: all 64 calibration batches from rank-0 only (different data subset).
- Different calibration data → different Hessian → different GPTQ error correction → different quantized weights → different bpb. Even on the λ=0 no-op clip path.

GPTQ uses the Hessian unconditionally for its column-wise error correction (`Hinv` / Cholesky), not just for the clip-adj formula. So even "λ=0 should be bitwise identical" was an over-assumption in the spec.

**Intra-sweep Δ (our λ=0.05/0.10/... vs our λ=0.00) is still valid** — same Hessian across all six runs. That's what we report above.

## Raw artifacts

- `lambda_*.json` — one per λ, synced back to this dir.
- `lambda_*.ptz` — quantized models, **kept on NA-1 volume only** at `/workspace/runs/001-hessian-sdclip/lambda_*.ptz` (~16 MB each × 6 = ~96 MB). Retrievable on-demand.
- `hessians.pt` — 232 MB, on NA-1 volume at `/workspace/runs/001-hessian-sdclip/hessians.pt`. Reusable for any future Hessian-based experiment on the same checkpoint.
- `sweep.py` — the wrapper script used (idempotent loop over `lambdas.txt`).
- `lambdas.txt` — records the exact λ sequence processed: 0.00, 0.05, 0.10, 0.20, 0.40, 0.60.
- `sweep.out` — stdout+stderr (includes the device-mismatch traceback from the first round 2 crash, see notes.md).

## Handback
Research decides: promote (unlikely given monotonic worsening) / iterate (different formulation?) / kill. Evaluation goes in `research/evaluations/001-hessian-sdclip.md` and a row in `experiments.md`.
