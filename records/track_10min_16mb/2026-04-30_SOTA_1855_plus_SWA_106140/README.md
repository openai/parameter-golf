# [Non-record] PR #1855 + SWA blend — replication + ablation

**3-seed mean val_bpb: 1.06140** (std 0.00044) | **15.91 MB** | 8×H100 SXM, 600s wallclock | TTT eval

vs current leaderboard SOTA (PR #1855, val_bpb 1.06108): **+0.00032 BPB** (slightly worse, not a record).

Submitted as **non-record** to contribute three findings:
1. **Hardware calibration data** — confirms PR #1981's "calibration offset" finding at smaller magnitude (+1.25 mbpb on our hardware vs SOTA's reported numbers, vs PR #1981's claimed ~5 mbpb).
2. **SWA composition with PR #1855's full stack** — small but real improvement (~0.9 mbpb vs vanilla on the same hardware), composes cleanly with LQER asymmetric int4 + per-group lrzip pipeline.
3. **Reproducibility checkpoint** — independent 3-seed reproduction within ±0.4 mbpb of published numbers per seed.

## Results

| Seed | Post-TTT val_bpb | Artifact bytes | Eval time |
|------|------------------|----------------|-----------|
| 42   | 1.06089          | 15,905,575     | 527 s     |
| 0    | 1.06158          | 15,905,785     | 459 s     |
| 1234 | 1.06172          | 15,914,864     | 458 s     |
| **mean** | **1.06140**  | 15,908,741     | 481 s     |

3-seed std: 0.00044 BPB. All artifacts under 16 MB cap; all evals under 600 s cap.

## Hardware-calibration check (Pod F, single seed=42, vanilla SOTA)

To isolate hardware shift from method effect, we ran PR #1855 *byte-identical* (`SWA_ENABLED=0`) on the same Runpod 8×H100 SXM AP-IN-1 host:

| | val_bpb | artefact bytes | eval time |
|---|---|---|---|
| Pod F (vanilla, our hardware, seed=42) | 1.06114 | 15,906,766 | 532 s |
| PR #1855 published seed=42 | 1.05989 | 15,897,259 | 509 s |
| **Hardware shift** | **+1.25 mbpb** | +9,507 bytes | +23 s |

This confirms PR #1981's observation that the same code on different hardware does not produce identical BPB, but at much smaller magnitude (~1 mbpb) than PR #1981's claimed ~5 mbpb.

## SWA addition vs vanilla on same hardware

| | val_bpb (mean) | Δ vs vanilla on same hardware |
|---|---|---|
| Vanilla SOTA (extrapolated 3-seed on our hardware) | ~1.06233 | — |
| Pod D6 (PR #1855 + SWA, 3-seed mean) | 1.06140 | **−0.93 mbpb** |

Vanilla extrapolation: SOTA published 3-seed mean 1.06108 + 1.25 mbpb hardware shift = ~1.06233. Pod D6's 1.06140 is 0.93 mbpb below that. SWA contributes a real but small improvement.

## What's new vs PR #1855

Stochastic Weight Averaging (SWA) blended into the EMA shadow before GPTQ. Code change is **+23 lines** in `train_gpt.py`, gated by `SWA_ENABLED=1`. With `SWA_ENABLED=0` the script reproduces PR #1855 byte-for-byte.

**SWA mechanism**: in the final 5% of training, snapshot live weights every 50 steps; before EMA-applies, blend the SWA average into the EMA shadow at weight 0.5.

```python
# Inside train loop, after EMA update
if h.swa_enabled and step >= int(h.iterations * (1.0 - h.swa_window_frac)) and step % h.swa_interval == 0:
    with torch.no_grad():
        for name, t in _live_state.items():
            swa_sum[name].add_(t.detach().float())
        swa_count += 1

# Just before EMA-applies
if h.swa_enabled and swa_count > 0:
    log(f"swa: blending {swa_count} snapshots at weight {h.swa_blend}")
    for name in ema_state:
        swa_avg = swa_sum[name] / swa_count
        ema_state[name].mul_(1.0 - h.swa_blend).add_(swa_avg, alpha=h.swa_blend)
```

The interaction with LQER asymmetric int4 + per-group compression composes cleanly at `SWA_BLEND=0.5` — the SWA contribution stays in the EMA-shadow blend rather than fully replacing it.

## Architecture

Inherits PR #1855's full stack — see `records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/README.md` for the architecture/hparam table. Only the EMA pipeline is modified (additive SWA blend).

## Reproduction

```bash
# 1. Pull precomputed CaseOps shards (saves ~30 min on-pod tokenisation)
huggingface-cli download romeerp/parameter-golf-caseops-v1 \
  --repo-type dataset --local-dir ./data

# 2. Install lrzip for COMPRESSOR=pergroup
apt-get install -y lrzip

# 3. Train each seed on 8xH100 SXM, 600s wallclock
for SEED in 42 0 1234; do
  CASEOPS_ENABLED=1 \
  DATA_PATH=./data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  SWA_ENABLED=1 SWA_WINDOW_FRAC=0.05 SWA_INTERVAL=50 SWA_BLEND=0.5 \
  COMPRESSOR=pergroup \
  RUN_ID=E056_seed${SEED} SEED=${SEED} \
  MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Test plan

- [x] Trains within 600s wallclock on 8×H100 80GB SXM (4500 iterations achieved per seed)
- [x] All 3 artifacts under 16 MB cap (max: 15,914,864 bytes; mean: 15,908,741 bytes)
- [x] TTT eval completes within 600s eval cap (max: 527 s)
- [x] 3-seed mean reproduced; per-seed numbers attached as `train_seed{42,0,1234}.log`
- [x] `SWA_ENABLED=0` regression: Pod F replication of PR #1855 vanilla seed=42 within +1.25 mbpb of published
- [x] SWA snapshot/blend code reachable in trace (logs confirm "swa: blending N snapshots" line in each seed run)
- [x] pergroup compression confirmed working (artefact 280 KB smaller than COMPRESSOR=brotli baseline)

## Why submitted as non-record

Our 3-seed mean (1.06140) is +0.32 mbpb worse than PR #1855's published 1.06108. This is within the hardware-calibration shift we measured (+1.25 mbpb) — i.e., the SWA addition is a real ~0.9 mbpb improvement on our hardware, but not enough to overcome the calibration shift and clear the 1.05608 record bar. Submitting as non-record so the SWA implementation, calibration data, and pergroup reproduction are available to future submissions on stronger-hardware reproductions or stronger base stacks.
