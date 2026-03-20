# Top-Heavy FFN + Packed Int6 Export

## Status

This folder is **PR-ready but eval-pending**.

We do **not** currently have GPU access, so there is no 8xH100 training log yet and no claimed leaderboard score. What is included here is:

- a standalone `train_gpt.py`
- a real CPU `DRY_RUN=1` smoke log
- exact local measurements for parameter count and export-size sanity checks

## Core Idea

The strongest public PRs are converging on the same lane:

- uniform `9 x 512` backbone
- wider `3x` FFN
- low-bit export
- sliding-window eval

This submission keeps the proven surrounding infrastructure, but changes **where the parameters live**.

Instead of a uniform `3x` FFN in every block, it uses **OpenELM-style layer-wise scaling** so later layers get larger FFNs and early layers get smaller ones, while keeping the same overall parameter budget:

`768, 960, 1152, 1344, 1536, 1728, 1920, 2112, 2304`

The motivation is simple:

- under a fixed artifact budget, later layers are closer to the loss and should benefit more from extra capacity
- early layers mostly need to build local lexical/syntactic features and can be narrower
- the 10-minute training budget makes it especially important to allocate parameters where gradients are most direct

## Design Choices

- **Top-heavy FFN allocation**
  - Primary novelty. Same total FFN budget as a uniform `3x` model, but shifted toward later layers.
- **Tied embeddings**
  - Baseline-efficient parameter sharing still matters at this scale.
- **Packed int6 export**
  - Large 2D matrices are stored as exact packed 6-bit values with per-row fp16 scales.
  - This avoids relying on an external `zstd` dependency during evaluation.
- **FP16 embedding passthrough**
  - The tied embedding / output head is the most quantization-sensitive tensor.
- **Sliding-window eval**
  - Included because it is legal under the rules and consistently improves token context at scoring time.
- **Tuned short-horizon training defaults**
  - Lower LR, higher Muon momentum, longer warmdown, and gradient clipping follow the public lessons from early strong submissions.
- **CPU dry-run mode**
  - `DRY_RUN=1` swaps in synthetic data and a tiny smoke-test architecture so contributors can verify the full train/export/eval path locally without CUDA.

## Local Measurements

Exact full-submission architecture:

- Parameter count: `21,778,504`
- FFN schedule: `768,960,1152,1344,1536,1728,1920,2112,2304`
- Code size: `60,837` bytes

Measured export-path sanity checks:

- Init export, packed int6 + zlib:
  - model bytes: `4,212,553`
  - total bytes: `4,273,390`
- Dense-random stress probe, packed int6 + zlib:
  - total bytes: `16,549,133`

Interpretation:

- The init export is a correctness check, not a realistic trained-artifact estimate.
- The dense-random probe is intentionally pessimistic.
- Final trained artifact size remains **pending compute**.

## Baseline Reference

Public baseline from the repo:

- Baseline `val_bpb`: `1.22436570`
- Baseline artifact size: `15,863,489` bytes

## Risks

- The layer-wise FFN schedule is well-motivated, but it has not yet been validated on this exact challenge distribution.
- Packed int6 + zlib is self-contained, but the final trained model still needs to demonstrate enough compressibility to stay comfortably below the cap.
- Because the current frontier already uses strong eval-time tricks, the gain from better parameter allocation may be modest unless it also improves quantization robustness.

## Reproducibility

Seed defaults to `42`.

CPU smoke test:

```bash
DRY_RUN=1 RUN_ID=topheavy_dryrun python train_gpt.py
```

Intended real run once compute is available:

```bash
RUN_ID=topheavy_ffn_packed_int6 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` — standalone trainer and exporter
- `submission.json` — pending-eval metadata
- `dry_run.log` — verified local 10-step CPU smoke run

## References

- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022
- Mehta et al., *OpenELM*, 2024
