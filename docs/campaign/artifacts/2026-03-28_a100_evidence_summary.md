# Pegasus Development Evidence Summary

Date: 2026-03-28
Hardware:
- NVIDIA A100-SXM4-80GB (`serv-3333`, `serv-3338`, Pegasus `A100-80GB`)
- NVIDIA H100 80GB HBM3 (`serv-3343`, Pegasus `H100`)
Purpose: Development evidence for compute grant application, plus first H100 verification

---

## Runs Completed

### Run 1: Smoke Test (200 iterations, reduced batch)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_baseline_smoke |
| Config | 9L 512d, batch 65536, 200 iter, 1 shard |
| Steps | 200/200 |
| Final post-quant val_bpb | 2.1737 |
| Step avg | 154.57 ms |
| Peak memory | 1548 MiB |
| Artifact int8+zlib | 7,066,088 bytes |

### Run 2: 600s Baseline (full batch, wallclock-capped)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_baseline_600s |
| Config | 9L 512d, batch 524288, 10 shards, 600s cap |
| Steps | 907/20000 |
| Final post-quant val_bpb | **1.3714** |
| Step avg | 661.65 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 12,046,627 bytes |
| Val BPB progression | 4.11 → 1.66 → 1.50 → 1.43 → 1.38 → 1.37 |

### Run 3: 600s LowerLR Comparison (controlled variant)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_lowerlr_600s |
| Config | Same as baseline, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 |
| Steps | 908/20000 |
| Final post-quant val_bpb | **1.3776** |
| Step avg | 661.00 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 10,723,611 bytes |
| Val BPB progression | 4.11 → 1.64 → 1.49 → 1.42 → 1.38 → 1.37 |

### Run 4: 600s Baseline Reproducibility Check (seed 42)

| Metric | Value |
|--------|-------|
| RUN_ID | a100_baseline_seed42_600s |
| Config | Same as baseline, `SEED=42` |
| Steps | 900/20000 |
| Final post-quant val_bpb | **1.3746** |
| Step avg | 667.26 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 12,018,778 bytes |
| Val BPB progression | 4.11 → 1.65 → 1.50 → 1.43 → 1.38 → 1.37 |

### Run 5: 600s Warmdown-Only Variant

| Metric | Value |
|--------|-------|
| RUN_ID | a100_warmdown3600_600s |
| Config | Same as baseline, `WARMDOWN_ITERS=3600` |
| Steps | 903/20000 |
| Final post-quant val_bpb | **1.4106** |
| Step avg | 665.18 ms |
| Peak memory | 10253 MiB |
| Artifact int8+zlib | 9,951,155 bytes |
| Val BPB progression | 4.11 → 1.72 → 1.52 → 1.45 → 1.41 → 1.40 |

### Run 6: 1xH100 600s Root Baseline

| Metric | Value |
|--------|-------|
| RUN_ID | h100_baseline_600s |
| Config | Same root baseline, 1xH100, batch 524288, 10 shards, 600s cap |
| Steps | 1795/20000 |
| Final post-quant val_bpb | **1.3059** |
| Step avg | 334.31 ms |
| Peak memory | 10303 MiB |
| Artifact int8+zlib | 14,684,525 bytes |
| Val BPB progression | 1.35 @1200 → 1.33 @1400 → 1.31 @1600 → 1.30 @1795 |

### Run 7: 8xH100 600s Root Baseline

| Metric | Value |
|--------|-------|
| RUN_ID | h100_8gpu_baseline_600s |
| Config | Same root baseline, 8xH100, Slurm-native `srun`, batch 524288, 10 shards, 600s cap |
| Steps | 11611/20000 |
| Final post-quant val_bpb | **1.2337** |
| Step avg | 51.66 ms |
| Peak memory | 10184 MiB |
| Artifact int8+zlib | 15,871,532 bytes |
| Val BPB progression | 1.31 @2600 → 1.29 @3600 → 1.28 @4000 → 1.24 @11000 → 1.23 @11611 |

---

## Controlled Comparison

| Metric | Baseline (default LR) | LowerLR | Delta |
|--------|----------------------|---------|-------|
| MATRIX_LR | 0.04 | 0.02 | -50% |
| SCALAR_LR | 0.04 | 0.02 | -50% |
| TIED_EMBED_LR | 0.05 | 0.03 | -40% |
| val_bpb | **1.3714** | 1.3776 | +0.0062 |
| Artifact size | 12.0 MB | 10.7 MB | -1.3 MB |

Interpretation: Default LR produces marginally better val_bpb. Lower LR produces a smaller artifact (lower weights = better compressibility). Both runs structurally identical in step count, memory, and stability.

## Reproducibility Check

| Metric | Baseline seed 1337 | Baseline seed 42 | Delta |
|--------|--------------------|------------------|-------|
| val_bpb | **1.3714** | 1.3746 | +0.0032 |
| Steps | 907 | 900 | -7 |
| Step avg | 661.65 ms | 667.26 ms | +5.61 ms |
| Artifact size | 12.05 MB | 12.02 MB | -0.03 MB |

Interpretation: The baseline remains materially stable across two seeds on 1xA100. The seed-42 run is slightly worse, but the spread is small enough to support a grant claim of reproducible operator behavior rather than one-off luck.

## Schedule Negative Control

| Metric | Baseline | Warmdown3600 | Delta |
|--------|----------|--------------|-------|
| WARMDOWN_ITERS | 1200 | 3600 | +2400 |
| val_bpb | **1.3714** | 1.4106 | +0.0392 |
| Steps | 907 | 903 | -4 |
| Step avg | 661.65 ms | 665.18 ms | +3.53 ms |
| Artifact size | 12.05 MB | 9.95 MB | -2.10 MB |

Interpretation: Extending warmdown alone is clearly harmful on this 1xA100 600s setup. It improves compressibility but hurts the actual objective. This is a useful negative control because it shows the pipeline is discriminating between plausible schedule changes rather than producing noisy ties.

## H100 Upgrade Read

| Metric | A100 baseline | 1xH100 baseline | Delta |
|--------|---------------|-----------------|-------|
| val_bpb | 1.3714 | **1.3059** | -0.0655 |
| Steps in 600s | 907 | 1795 | +888 |
| Step avg | 661.65 ms | 334.31 ms | -327.34 ms |
| Artifact size | 12.05 MB | 14.68 MB | +2.64 MB |
| Peak memory | 10253 MiB | 10303 MiB | +50 MiB |

Interpretation: The exact same root baseline improves materially moving from 1xA100 to 1xH100. This is the strongest hardware evidence collected so far because it is closer to the challenge target while remaining under the 16 MB artifact cap.

## 8xH100 Upgrade Read

| Metric | 1xH100 baseline | 8xH100 baseline | Delta |
|--------|------------------|-----------------|-------|
| val_bpb | 1.3059 | **1.2337** | -0.0723 |
| Steps in 600s | 1795 | 11611 | +9816 |
| Step avg | 334.31 ms | 51.66 ms | -282.65 ms |
| Artifact size | 14.68 MB | 15.87 MB | +1.19 MB |
| Peak memory | 10303 MiB | 10184 MiB | -119 MiB |

Interpretation: Moving from `1xH100` to `8xH100` with the same root baseline and fixed 600-second wallclock cap produces another large gain, bringing the root baseline to `1.2337` post-roundtrip while still remaining under the challenge artifact cap. This is the first true challenge-shaped result in this campaign.

## 8xH100 Launch Lesson

Date: 2026-03-28
Node: `serv-3342`
Allocation: `8xH100 80GB HBM3`

Observed behavior:

- All 8 GPUs are visible in the Slurm allocation.
- `torchrun --standalone --nproc_per_node=8 train_gpt.py` never prints the initial logfile path.
- A minimal 8-rank NCCL smoke (`/tmp/nccl_test.py`) under `torchrun --standalone` also hangs before any rank output.
- `torch.distributed.elastic` reports `RendezvousTimeoutError`.
- Re-allocating with `--ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6` and launching with `srun` succeeds.
- A minimal 8-rank NCCL smoke under Slurm-native `srun` prints all `rank X ok` and `rank X barrier ok` lines.
- The full trainer then runs successfully under the same Slurm-native launch shape.

Interpretation:

- The blocked path is specifically `torchrun --standalone` rendezvous on this Pegasus node.
- This was not evidence of a bug in `train_gpt.py`.
- The correct launch pattern for Pegasus `8xH100` work in this campaign is Slurm-native `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none`, not `torchrun --standalone`.

---

## What This Demonstrates

1. **End-to-end pipeline works:** Training, evaluation, int8 quantization, zlib compression, and post-quantization round-trip validation all execute correctly on Pegasus.
2. **AMP dtype auto-detection works:** bf16 selected automatically on A100.
3. **Controlled experimentation capability:** Baseline vs variant comparison with only LR changed, a baseline seed-repeat for reproducibility, and a clearly negative warmdown-only schedule test.
4. **Hardware scaling evidence:** The same root baseline improves significantly on 1xH100 relative to 1xA100, and again on 8xH100 relative to 1xH100.
5. **Challenge-shaped execution:** The root baseline now runs successfully for the full 600-second budget on real `8xH100`.
6. **Artifact fits challenge budget:** All measured runs remain under the 16 MB cap, including the `8xH100` run at `15,871,532` bytes.
7. **Operator readiness:** Dataset download, environment setup, distributed launch debugging, and training execution completed without external assistance on shared HPC infrastructure.

## What This Does Not Demonstrate

- A competitive top-tier architecture
- Competitive val_bpb (baseline 1.37 on 1xA100 vs leaderboard 1.22 on 8xH100)
- Advanced techniques (int6, XSA, EMA, TTT, etc.)

## Hardware Note

These runs used `1xA100-SXM4-80GB`, `1xH100 80GB HBM3`, and `8xH100 80GB HBM3`. The challenge leaderboard target is `8xH100-SXM5`. This summary therefore demonstrates development readiness, reproducibility, first H100 verification, and a real challenge-shaped root baseline on Pegasus hardware. It still does not demonstrate a competitive top-tier architecture; H100 multi-GPU time is now justified for actual model improvements rather than basic infrastructure debugging.

---

## Grant Application Support

This evidence package now supports an Intermediate-level compute grant request:
- Demonstrates working infrastructure and operator competence
- Shows controlled experimental methodology
- Shows small seed sensitivity on the current baseline
- Shows that plausible schedule changes can be rejected cleanly with current evidence
- Shows that the root baseline improves materially on H100 hardware and in full `8xH100` challenge-shaped execution
- Provides a clear path from current state to competitive submission
- H100 multi-GPU time is now justified for actual model changes and the pre-TTT anchor port, not for baseline infrastructure bring-up
