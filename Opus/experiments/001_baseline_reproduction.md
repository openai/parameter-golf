# Experiment 001 — SOTA reproduction (seed=42)

**Date:** TBD (Day 1)
**Hypothesis:** We can reproduce PR #1493's seed-42 result `val_bpb_ttt = 1.08079` to within ±0.0003 on 1×H100. Reproduction is a hard prerequisite before any modification — a number that doesn't reproduce isn't a leaderboard-valid base.
**Baseline:** PR #1493 published `val_bpb_ttt = 1.08079` (seed 42, 8×H100 SXM)
**Cost:** ~2h × $3/hr = ~$6 of credits

## Setup

```bash
# On a 1×H100 RunPod pod (template y5cejece4j)
cd /workspace
git clone https://github.com/GodlyDonuts/parameter-golf.git
cd parameter-golf
git checkout claude/busy-thompson-9c94f9   # the branch with the Opus folder

# Install brotli + flash-attn-3 (FA3 wheel matches the SOTA's python 3.12 / cu128 image)
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download data — sp8192 variant
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

## Config

We're running the published SOTA `train_gpt.py` exactly as submitted, no modification. This is the LZMA-wrapped one-liner version.

```bash
cd /workspace/parameter-golf

SEED=42 \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  QK_GAIN_INIT=5.25 \
  RUN_ID=opus_repro_seed42 \
  torchrun --standalone --nproc_per_node=1 \
    records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py
```

**Note — `nproc_per_node=1`:** on 1×H100 we lose ≈8× the per-step grad-accum. The SOTA uses `grad_accum_steps=8//world_size`, so on 1×H100 we accumulate 8 micro-batches. The 600s wallclock cap will hit at fewer steps than the published 4550 — we accept this for reproduction; what we want to confirm is the architecture/data pipeline is correctly wired.

## Result

| Metric | Expected | Actual | Notes |
|--------|----------|--------|-------|
| `val_bpb` (pre-quant) | ~1.07 | | sanity check |
| `val_bpb_quantized` | ~1.085 | | post-GPTQ |
| `val_bpb_quantized_sliding` | ~1.0827 | | sliding window eval |
| `val_bpb_quantized_ttt` | **~1.0808** | | the SOTA number |
| Wallclock train | ≤ 600s | | should hit cap |
| Wallclock eval | ≤ 600s | | TTT eats most of this |
| Artifact bytes | ~15.99M | | should be under 16M |

Tolerance: 1×H100 may produce a slightly different number than 8×H100 due to grad-accum ordering and float16 reductions. Acceptance bar:
- Pre-quant val_bpb within ±0.005 of expected
- Final TTT val_bpb within ±0.005 of 1.08079

If outside that range — investigate. **Do not** proceed to Day 2 sweeps until reproduction is in tolerance.

## Decision

- ✅ **Within tolerance** → proceed to experiment 002 (selective-TTT comparison). Save the final checkpoint (the `final_model.int6.ptz` artifact) — Day 2 reuses it.
- ⚠️ **Off by 0.001–0.005** → re-run on 8×H100 to rule out the 1-GPU-vs-8-GPU effect; if the 8-H100 number reproduces, accept.
- ❌ **Off by >0.005** → debug (data version? Python version? FA3 version? GPTQ randomness?). Stop spending compute until resolved.

## Save the checkpoint

After the run, before the pod terminates:

```bash
# Inspect the artifact files written by the run
ls -lh final_model*

# Pull the checkpoint to local for Day 2 reuse
runpodctl send <pod-id>:/workspace/parameter-golf/final_model.int6.ptz ./Opus/artifacts/seed42.int6.ptz
# (alternative: scp via the pod's SSH endpoint)
```

The `.ptz` is the brotli-compressed int6 quantized model — about 16MB. Day 2 experiments load it directly via `deserialize()` and skip the 600s training step entirely.
