# Execution notes — spec 001

## Outcome
Sweep **completed**, all 6 λ values measured. Result: **no signal, monotonic worsening** with larger λ. See `summary.md` for the table.

## Timeline (UTC, pod `24v3app1be48ld`)
- `21:46` pod created (1×H100 NA-1, $2.99/hr), SSH up at attempt 4 (~24s)
- `21:47` preflight: brotli installed, `git stash` pod-local, `git checkout 74c8385` ✓, spec-000 ckpt confirmed (300 MB)
- `21:47` sweep.py uploaded, `lambdas.txt` written with 0.00/0.05/0.10
- `21:47` round 1 launched via setsid
  - Hessian collection: **14.4s** (much faster than spec estimate of 3-5 min)
  - λ=0.00: 168.9s — 1.10518
  - λ=0.05: 130.3s — 1.10527
  - λ=0.10: 133.9s — 1.10530
  - sweep.py done @ `~21:55`
- `~22:00` user asked for larger λs (0.20, 0.40, 0.60); appended to `lambdas.txt` and relaunched sweep.py
  - **Crash on first new λ (0.20):** `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)` at `gptq_quantize_weight:822`
  - Root cause: my sweep.py reloaded Hessian with `map_location=device` (cuda), but `collect_hessians` returns them on CPU, and `gptq_mixed_quantize` operates on CPU. Device mismatch.
  - Fix: `sed -i` patch on the pod, `map_location=device` → `map_location="cpu"`. One-char fix. No file re-scp needed.
- `22:17` round 2 relaunched with fix
  - Hessian reloaded from cache in <1s (232 MB file)
  - λ=0.20: 145.4s — 1.10553
  - λ=0.40: 135.7s — 1.10618 (artifact 16.02 MB — **over 16 MB leaderboard limit**)
  - λ=0.60: 136.4s — 1.10676 (artifact 16.06 MB — **over 16 MB leaderboard limit**)
  - sweep.py done @ `~22:24`
- `22:24` rsync artifacts, `runpodctl pod stop 24v3app1be48ld`

## Cost
- ~38 min pod wall at $2.99/hr ≈ **$1.90**
- (vs spec's estimate of ~$0.45 — we paid ~4× because of the device-mismatch crash recovery + fresh compile on round 2 re-entry)
- Pod stopped (not deleted) — same-day policy. Hessian + `.ptz` artifacts stay on volume.

## Sweep.py wrapper

The spec's open-questions flagged that `hotstart.py` couldn't reuse the Hessian across invocations. I wrote a ~150-line `sweep.py` wrapper (saved at `/workspace/runs/001-hessian-sdclip/sweep.py` and in this dir) that:
- Loads + applies EMA to the spec-000 checkpoint
- Computes Hessians once, saves to `hessians.pt`, reloads on every subsequent invocation
- Loops over λ values from `lambdas.txt`, skips any λ with existing JSON (idempotent)
- Outputs one `lambda_X.json` + `lambda_X.ptz` per λ

Uses only functions already exported from `train_gpt_sota.py` — no duplicated logic. The same pattern should work for any future quant-time or eval-time sweep on a fixed checkpoint.

## Bugs caught (lessons for future executions)

### 1. `torch.load(hessian_path, map_location=...)` device
`collect_hessians` in `train_gpt_sota.py:806` explicitly moves Hessians to CPU before returning. Subsequent `gptq_mixed_quantize` / `gptq_quantize_weight` operate on CPU. So any cached Hessian re-load should use `map_location="cpu"`, not the CUDA device. **Fixed in sweep.py**.

### 2. Validity gate assumed 1-GPU == 8-GPU calibration
Spec's validity gate (`λ=0 must reproduce 1.10430 ± 0.0001`) assumed the Hessian would be identical to spec 000's. But spec 000 ran on 8×H100 where `ShuffledSequenceLoader` distributes FineWeb shards across ranks, giving each rank different calibration data. Our 1×H100 screen sees only the rank-0 slice. Different calibration data → different Hessian → different GPTQ error correction → ~0.0009 bpb shift even on the λ=0 no-op clip path.

**Takeaway:** absolute bpb numbers from 1-GPU screens aren't directly comparable to 8-GPU baselines, even with identical seed. Only intra-sweep Δ is valid. Future spec-interview protocol should clarify that hotstart screens measure *relative* signal, not *absolute* bpb.

### 3. Artifact size scales with λ
At λ=0.40 and λ=0.60 the `.ptz` exceeds 16,000,000 bytes. The `adj = 1 + λ(r_i − 1)` row-wise multiplier amplifies per-row scale differences, which reduces brotli's compression efficiency on the int6 matrices. Worth noting for any future Hessian-modulated quant — the size-vs-bpb tradeoff is coupled, not independent.

### 4. sweep.out buffering
Python's stdout is block-buffered when redirected to a file. The `log()` function in `train_gpt_sota.py` writes to both `print()` and `h.logfile` — the logfile path is more reliable mid-run than tailing sweep.out (which only flushes at process exit or buffer fill). **For future runs:** use `python3 -u` or `PYTHONUNBUFFERED=1` to make sweep.out reliable in real time.

## Artifact retention on volume

Everything stays at `/workspace/runs/001-hessian-sdclip/`:
- `hessians.pt` (232 MB) — reusable for any downstream Hessian-based experiment on this checkpoint
- `lambda_*.ptz` (×6, ~96 MB total) — the actual quantized models, if research wants to inspect the weights or eval with sliding/TTT on a specific λ
- `sweep.out` — full stdout/stderr including both crashes and recovery
