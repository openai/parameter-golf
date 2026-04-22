# Spec 016 seed_42 screen — execution notes

**Pod:** `w20na0rp2710e8` (JP 8×H100), stopped after rsync.
**Commit:** `4dd2d63` (α=1 init + grad_norm pre-step snapshot fix).
**Env:** standard #1736 screening env + `RECUR_ALPHA_ENABLED=1` + `TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache` (first run, populates).

## Result at a glance

| measurement | 008 | 015 | 016 | 016 Δ vs 015 |
|---|---|---|---|---|
| stopping step | 4828 | 4761 | **4708** | −53 |
| training wallclock | 596 s | 596 s | 596 s | = |
| endpoint val_bpb | 1.0697 | 1.0696 | **1.0712** | **+0.0016** |
| matched-step val_bpb @ 4000 | 1.1110 | 1.1078 | **1.1072** | **−0.0006** |

**Endpoint interpretation:** 016 ran 53 fewer steps than 015, entirely attributable to JP pod hardware variance (tok/s ~1.1% below 015's pod late in training). **Matched-step @ 4000 is the honest read.**

**Matched-step Δ vs 015 = −0.0006 → null bucket** per spec 016 decision criterion. α-init doesn't meaningfully change screen-level outcome: the learnable-α mechanism produces ~the same training dynamics whether initialized at 0 or 1.

## α trajectory — 015 vs 016

Both runs converge to similar plateau shape; 016 runs ~0.10 "hotter" everywhere:

```
step  | 015 α                                       | 016 α
------+---------------------------------------------+---------------------------------------------
 2000 | [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]    | [[1.00, 1.00, 1.00], [1.00, 1.00, 1.00]]   init
 2200 | [[0.03, 0.07, 0.14], [0.16, 0.24, 0.33]]    | [[0.84, 1.02, 0.90], [0.75, 0.76, 0.88]]   post-act
 3000 | [[1.04, 1.16, 1.38], [0.98, 0.86, 0.76]]    | [[1.13, 1.30, 1.40], [1.04, 0.93, 0.85]]
 4000 | [[1.04, 1.16, 1.38], [1.01, 0.89, 0.77]]    | [[1.13, 1.30, 1.40], [1.04, 0.96, 0.85]]
 4700 | [[1.04, 1.16, 1.38], [1.01, 0.89, 0.77]]    | [[1.13, 1.30, 1.40], [1.04, 0.96, 0.85]]   saturated
```

Same directional structure (pass-2 increasing with depth, pass-3 decreasing). 016 ending α is ~0.10 higher than 015's everywhere.

## grad_norm fix works

Spec 015 always logged α grad_norm=0.0 (log fired after zero_grad). Commit 4dd2d63 snapshots pre-step. This run's grad_norm post-activation is 0.001–0.007 range. Plumbing confirmed.

## torch.compile cache (spec 016 onward)

Commit 4dd2d63 compile+autotune cache lives at `/workspace/.torch_inductor_cache` on JP volume (~10 GB). Same-commit rerun on JP drops 10 min compile → ~1–2 min. Different commit or different region = no reuse.

## Timeline

| phase | wall |
|---|---|
| pod create + SSH | ~1 min |
| compile + CUDA-graph bucket warmup | ~9 min (first run, fills cache) |
| training (596s wallclock cap) | ~10 min (step 4708 endpoint) |
| post-train serialize | ~15 s (`final_model.pt` saved before user-kill) |
| rsync + pod stop | <1 min |
| **Total pod time** | **~20 min** |

## Cost

JP 8×H100 ~20 min × $23.92/hr ≈ **$8**.

## Deliverables

**Local (rsynced):**
- `train.log` (81 KB), `final.json`, `notes.md`, `launch.out`

**On JP volume (`jlxvxeiol4`, not rsynced):**
- `final_model.pt` (135 MB, pre-GPTQ FP32 post-EMA) — reusable as hotstart or for post-hoc GPTQ/TTT eval
- `/workspace/.torch_inductor_cache` — torch.compile cache keyed on 4dd2d63

## Handback

Per spec 016 decision criterion (Δ @4000 vs 015): **−0.0006 is null** → α-init does not change screen outcome; shelve init-experiments unless there's a reason to probe plateau shape differently. Research may want a matched-clock rerun to remove the 53-step ambiguity (~$8) but probably not worth it given null result. `final_model.pt` on JP is available for post-hoc follow-ups.

## Post-hoc TTT/GPTQ eval (appended 2026-04-21, ~$4)

**Pod:** `1dqkead11ztrxi` (JP 8×H100), stopped after rsync. **Method:** EVAL_ONLY_CHECKPOINT bypass patch in train_gpt.py — skips training, loads `final_model.pt`, runs GPTQ + diagnostics + phased-TTT.

### Numbers captured before crash

| stage | 016 val_bpb | 008 / #1736 equiv | Δ |
|---|---|---|---|
| pre-quant post-EMA | **1.07083** | 008: 1.06922, #1736: 1.06906 | +0.0016 vs 008 |
| post-quant (GPTQ int6) | **1.08029** | #1736: 1.07847 | +0.0018 vs #1736 |
| artifact size | **15.94 MB** | #1736: 15.98 MB | −0.04 MB (under 16 MB cap) |
| quantized + TTT | **crashed OOM** | #1736: 1.06610 | — |

**Quantization cost:** 016 = +0.00947, #1736 = +0.00941. Quantization behavior is unchanged by α mechanism.

### OOM details

Multiple ranks OOM'd during `eval_val_ttt_phased` — `CUDA out of memory. Tried to allocate 960 MiB. GPU X has 528 MiB free.` All 8 GPUs oversubscribed. Cause: my EVAL_ONLY_BYPASS patch skipped train_model's warmup phases (CUDA graph bucket warmup, loop_warmup), which normally prime the CUDA caching allocator and run-through the ttt code paths once. Without that priming, the TTT eval's peak memory footprint exceeded what the allocator had pre-reserved.

**Not a correctness issue.** The pre-quant and post-quant numbers are good. Only the TTT stage's memory tuning depends on the warmup. A proper (full) rerun with training would get the TTT number cleanly.

### Projection (if TTT had run)

Apply #1736's observed TTT recovery (−0.01237) to our 1.08029: **projected post-TTT ≈ 1.06792**.
- **Inside the accept gate** (1.06310, 1.06910). ✓
- **+0.00182 worse than #1736's record** (1.06610). Not a clear submission-track win.

### Conclusion

Even with optimistic TTT projection, spec 016 (α=1 init) on this pod does not beat #1736's 1.06610. The gap is mostly the endpoint step-deficit from JP hardware variance (53 steps short of 015, 120 steps short of 008). On a hypothetical matched-step rerun (step 4828), projected post-TTT would drop ~0.0015 more (to ~1.06642), still not clearly beating #1736.

Combined with the null result vs 015 at matched step (Δ = −0.0006), the honest conclusion is: **recur-alpha mechanism gives a real but small (~0.003 val_bpb matched-step) improvement over spec 008 baseline, largely absorbed by TTT, and the α=0 vs α=1 init choice is irrelevant.** Not a submission promotion candidate on its own.

### Artifacts

Local: `runs/016-recur-alpha-ones/posthoc/posthoc.log` (full log including crash), `launch.out`.

On JP volume (preserved for future research): `final_model.pt` (130 MB), `final_model.int6.ptz` (15.94 MB, **already GPTQ'd and brotli-compressed** — saved this run before the TTT crash), `/workspace/.torch_inductor_cache` (~10 GB, 4dd2d63 graph hash).

### Incremental cost

| item | cost |
|---|---|
| JP 8×H100 pod for post-hoc eval (~12 min wall) | ~$4 |
| **Spec 016 grand total** (screen + post-hoc) | **~$12** |
