# Spec 002 — SWA + EMA blend screen

**Slug:** `swa-plus-ema`
**Created:** 2026-04-19
**Links to idea:** `research/ideas/swa-plus-ema.md`

## Hypothesis
Uniform averaging of the model's post-recurrence warmdown snapshots (pure SWA), and blends of that SWA state with the existing EMA state, land on a wider/flatter minimum than EMA alone. Expected mechanisms: (1) better generalization on val, (2) better quantization robustness (quant penalty currently costs ~0.011 bpb — any fraction of that we reclaim is direct record-push ammunition).

## Baseline
This screen's own C0 (EMA-only) is the in-sweep baseline. It should land close to spec 001's λ=0 result of **1.10518** (NOT spec 000's 1.10430 — the 1×H100 Hessian shard differs from 8×H100's, causing a ~+0.0009 offset; see spec 001's summary). All C1-C5 Δ is measured against **this run's C0**, not against spec 000 or spec 001.

## Expected Δ
+0.0005 to +0.0015 bpb for the best blend config, with a realistic floor of +0 (SWA has been productive in low-bit-width quant literature, but SOTA's training + EMA are already well-tuned). Post-Hessian-SDClip-null-result priors are cool; don't count on a win.

## Accept criteria
- **Validity gate:** C0 must land at 1.10518 ± 0.0005 (loose because 1×H100 Hessian is non-deterministic at 5th decimal across pods). If C0 is materially different, pause and investigate before continuing.
- **Signal gate:** at least one non-control config with **Δ_quant ≤ −0.0003 AND Δ_sliding ≤ −0.0003** vs C0. Requiring both stages to move catches false positives where SWA helps raw quant but loses at sliding.

## Config diff
No hyperparam changes. The 6 configs are built inside `swa_sweep.py`:

| ID | Description | Sources |
|---|---|---|
| C0 | EMA-only (control) | load ema_state from `ckpt_final_pre_ema_step3849.pt` |
| C1 | SWA all 4 post-recurrence | uniform mean of {1500, 2275, 3412, 3849} raw weights |
| C2 | SWA late 3 | uniform mean of {2275, 3412, 3849} raw weights |
| C3 | 0.5·C1 + 0.5·EMA | blend |
| C4 | 0.25·C1 + 0.75·EMA | lean EMA |
| C5 | 0.75·C1 + 0.25·EMA | lean SWA |

**Pre-recurrence snapshots ({455, 1048, 1137, 1378}) are intentionally excluded.** The computational graph flips at step 1378 when `looping_active` toggles; averaging across that boundary pulls toward a non-recurrent minimum.

Env at launch: match spec-000 defaults — `BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=0 SEED=1337`. TTT explicitly disabled (screen is quant + sliding only; TTT saved for winners).

## Code changes
- Branch: `exp/swa-plus-ema`
- Commit: `46c2a92`
- Diff: new file `swa_sweep.py` (~260 lines), self-contained, imports existing machinery from `train_gpt_sota.py` (`collect_hessians`, `gptq_mixed_quantize`, `eval_val`, `eval_val_sliding`, etc.). No changes to `train_gpt_sota.py`.

## Hardware ladder
- [ ] 1×H100 NA-1 — **primary rung** (GPTQ is sequential per-matrix; no useful GPU parallelism for this workload).
- [ ] 2×H100 — fallback only if 1×H100 has memory pressure on calibration.
- [ ] 8×H100 — not used.

## Seed plan
Single seed (`SEED=1337`, matches spec 001's script default). The screen is deterministic modulo calibration-data sampling given fixed weights; seed variance within a config is small.

## Inputs
- **Hotstart checkpoints** (4 post-recurrence snapshots from spec 000):
  - `/workspace/runs/000-sota-replication/checkpoints/ckpt_event_step1500.pt`
  - `/workspace/runs/000-sota-replication/checkpoints/ckpt_event_step2275.pt`
  - `/workspace/runs/000-sota-replication/checkpoints/ckpt_event_step3412.pt`
  - `/workspace/runs/000-sota-replication/checkpoints/ckpt_final_pre_ema_step3849.pt` (also source of EMA state for C0/C3/C4/C5)
- Data: `/workspace/data/datasets/fineweb10B_sp8192/` (for Hessian calibration + val eval).
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`.
- Base repo commit: `46c2a92` on `exp/swa-plus-ema`.

## Execution protocol
Single invocation, single pod launch, all 6 configs in one script run:

```bash
HESSIAN_CLIP_LAMBDA=0 BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=0 SEED=1337 \
torchrun --standalone --nproc_per_node=1 swa_sweep.py \
  --ckpt_dir /workspace/runs/000-sota-replication/checkpoints \
  --run_dir /workspace/runs/002-swa-plus-ema
```

Script behavior:
1. Loads the 4 snapshots, extracts ema_state from the final one.
2. Builds all 6 config state-dicts upfront (fast, CPU-only averaging).
3. Computes Hessian ONCE from C0 weights, caches to `hessians.pt` on the run dir.
4. Loops over configs C0 → C1 → C2 → C3 → C4 → C5:
   - Load averaged state into GPT model.
   - GPTQ quantize (shared hessians).
   - Brotli compress → save `quantized_{CID}.ptz` on NA-1 volume.
   - `val_bpb_quantized` eval.
   - `val_bpb_sliding` eval.
   - Write `config_{CID}.json` with both numbers + metadata.
5. Idempotent: if `config_{CID}.json` exists, skip that config.

**Hessian reuse caveat (known simplification):** Hessian technically depends on weights via activations. Different averaged weights → slightly different Hessian. Using C0's Hessian for C1-C5 is a screening-stage approximation. If a config looks promising enough to promote, re-run it with a per-config Hessian to confirm the Δ holds.

## Checkpoints to emit
No training checkpoints. Per-config quantized artifacts (for cheap follow-up sliding/TTT eval if promoted):

- `/workspace/runs/002-swa-plus-ema/quantized_{C0..C5}.ptz` — ~16 MB each, ~100 MB total.
- `/workspace/runs/002-swa-plus-ema/hessians.pt` — ~232 MB, reusable for any future Hessian-based experiment on these checkpoints.
- Retention: keep through record-track push (2026-04-30).

## Stop-early criteria
- **Validity fail:** if C0 is outside 1.10518 ± 0.0005 by a wide margin → investigate before continuing. Possible causes: (a) averaging bug, (b) state-dict dtype mismatch, (c) wrong EMA source.
- **No-signal fail:** if C0, C1, C2, C3 are all within ±0.0002 of each other at both quant AND sliding → SWA doesn't transfer. Skip C4/C5. Save ~35% of remaining budget.
- **Standard:** NaN / obvious failure → kill and mark failed.

## Cost estimate
- 1×H100 NA-1 at ~$2.50/hr.
- Setup (pod + volume + deps + ckpt load): ~5 min.
- Hessian collection (one-time): ~3-5 min.
- Per config (GPTQ ~90s + quant eval ~60s + sliding eval ~3 min + save ~5s): **~5 min each × 6 = ~30 min**.
- **Total wall: ~40 min. Cost: ~$1.70.**
- Early-kill after validity fail: ~$0.60.
- Early-kill after no-signal (stop at C3): ~$1.20.

## Extra artifacts
- One JSON per config at `runs/002-swa-plus-ema/config_{CID}.json`: `{config_id, description, meta, val_bpb_quantized, val_loss_quantized, val_bpb_sliding, val_loss_sliding, artifact_size_bytes, elapsed_quant_sec, elapsed_sliding_sec}`.
- `runs/002-swa-plus-ema/summary.md` — aggregate table (all configs + Δ vs C0 at both stages). **Primary artifact for research evaluation.**
- `runs/002-swa-plus-ema/notes.md` — execution narrative.
- `runs/002-swa-plus-ema/sweep.out` — stdout+stderr capture.

No train.log (no training).

## Open questions for interview
- Confirm the 4 checkpoint files still exist on NA-1 volume at the expected paths.
- Confirm `/workspace/parameter-golf/` (or equivalent) has the `exp/swa-plus-ema` branch checked out at commit `46c2a92`.
- Confirm `torchrun --standalone --nproc_per_node=1` is the right invocation for 1×H100 (single-GPU rank 0). Spec 001 used this exact pattern successfully.
- Confirm ≥ 350 MB free on NA-1 volume for hessians.pt + 6 quantized artifacts.
- Clarify: if C0 validity fails, does execution auto-kill and ask research, or continue running C1-C5 for diagnostic data? Recommend: **stop and surface** — if C0 is off, all other Δ are still interpretable relative to C0 itself, so continuing is an option, but we want research eyes on it before proceeding.
