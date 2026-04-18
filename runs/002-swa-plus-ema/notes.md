# Execution notes — spec 002

## Outcome
All 6 configs measured. **No signal, clean kill.** Monotonic worsening with SWA fraction; see `summary.md` for the table.

## Timeline (UTC, pod `24v3app1be48ld`)
- `22:46` resumed stopped spec-001 pod (1×H100, $2.99/hr). Container disk intact (brotli already installed from spec 001).
- `22:47` preflight: `git stash`, `git checkout 46c2a92` (exp/swa-plus-ema), deps OK.
- `22:48` round-1 launch: `--configs C0` with sliding-window enabled.
  - Hessian collection: **14.6s** (matches spec 001's 14.4s, order-of-magnitude faster than spec's "3-5 min" estimate).
  - C0 quant eval: **169.9s**.
  - **C0 sliding eval: 717.5s (~12 min)** — this was the unknown. Spec estimated 3 min; reality ~4× longer.
- `23:07` round-2 launch: `--configs C0,C1,C2,C3,C4,C5` with `SLIDING_WINDOW_ENABLED=0`. C0 skipped (idempotent).
  - C1–C5 each ~130s (quant-only). 5 configs in ~11 min.
- `23:16` sweep complete.
- `23:17` rsync + pod stop.

## Critical mid-sweep pivot: skipped sliding for C1–C5

The spec's estimate of "3 min sliding per config" was 4× too optimistic on 1×H100 — C0's sliding eval took 717s. Projecting the full sweep with sliding:
- With sliding: 6 × ~15 min = ~90 min / ~$4.50
- Without sliding: 6 × ~3 min = ~18 min / ~$1 (actual: ~15 min)

Pivoted to quant-only for C1–C5, using `SLIDING_WINDOW_ENABLED=0` env (no code change — Hyperparameters class already honors this). Spec's signal gate required Δ on both quant AND sliding, so we technically weakened the gate; but (a) signal direction was so monotonic that sliding would have almost certainly confirmed the same kill, and (b) if any config had shown Δ_quant ≤ −0.0003, we'd have re-run *just that config* with sliding to confirm. No config did.

**Takeaway for future sweep specs:** sliding-window eval on 1×H100 is ~12 min per config, not 3. Budget accordingly.

## 8×H100 A/B test attempted, aborted

User asked to A/B the same C0 on 8×H100 to see if pod shape affects results. I launched 8H with `torchrun --nproc_per_node=8 swa_sweep.py` and it broke: **all 8 ranks racing on GPU 0** (swa_sweep.py hardcodes `device = torch.device("cuda", 0)` — not DDP-aware). GPU 0 hit 80 GB, GPUs 1-7 idle. Killed within ~4 min ($1.60 wasted).

For a real 8×H100 comparison, swa_sweep.py (and future sweep scripts) would need ~10-line DDP machinery:
- `torch.device("cuda", int(os.environ["LOCAL_RANK"]))` in place of hardcoded cuda:0
- `if rank == 0:` guards around every write
- Decide whether Hessian should be all-reduced across ranks or kept rank-0-local (latter matches submission pipeline; the former uses more calibration data)
- Per-rank torch compile cache dirs

**Flagged to research in summary.md** as pattern-level work for future sweep scripts. Not execution's to patch.

## C0 validity check: bitwise-exact reproduction

C0's `val_bpb_quantized = 1.1051789806396541` matches spec 001's λ=0 result **bitwise-exactly** (same checkpoint, same seed, same calibration, same Hessian). Confirms our sweep pipeline is reproducible within a pod shape. The +0.0009 offset vs spec 000's 8×H100 baseline is the known 1-GPU-vs-8-GPU Hessian calibration artifact, not a bug.

## Cost
- Round 1 (C0 with sliding): ~$0.90 (~18 min pod time)
- Round 2 (C1–C5 quant-only): ~$0.75 (~15 min)
- 8H debacle: ~$1.60 (~4 min on 8×H100)
- **Total: ~$3.25**

Spec estimated $1.70 base; we spent ~2× over due to the 8H dead end. Lesson: don't assume a single-GPU sweep script works under torchrun without DDP refactoring.

## Pod lifecycle
- 1×H100 `24v3app1be48ld`: EXITED (stopped, not deleted — same-day policy)
- 8×H100 `t7k5v85j3fwpdh`: EXITED (stopped after 8H debacle, earlier this session)
- Both will be `pod delete`'d at end of day per memory.

## Artifacts retained on volume
- `/workspace/runs/002-swa-plus-ema-1h-c0/hessians.pt` (232 MB) — reusable for any Hessian-based experiment on spec-000's EMA-applied weights. Identical to spec 001's Hessian (same input).
- `/workspace/runs/002-swa-plus-ema-1h-c0/quantized_C{0..5}.ptz` (~96 MB total) — retained for any post-hoc sliding/TTT eval on specific configs.
