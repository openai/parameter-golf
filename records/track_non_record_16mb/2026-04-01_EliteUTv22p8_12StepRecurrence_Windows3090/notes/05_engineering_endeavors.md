## 05 — Engineering Endeavors

This section documents the engineering effort behind the BPB improvements.

## Platform adaptation (Windows + 3090)

- Introduced `train_gpt_windows.py` launcher with backend-safe runtime patching.
- Hardened SDP backend behavior to avoid known Windows instability paths.
- Added compatibility guards around distributed/runtime behavior.

## Training reliability and reproducibility

- Added detailed config banners and run-level logging.
- Added explicit data seeding traceability (`RUN_ID`, `DATA_SEED`, deterministic flags).
- Added EMA checkpoint management and “best checkpoint by val_bpb” export policy.

## Memory and throughput engineering

- Added VRAM snapshots around eval-time EMA swap.
- Reworked eval swap path to avoid heavy transient GPU allocations.
- Preserved high-token global batch behavior with micro-batch accumulation controls.

## Optimization and stability stack

- Muon + AdamW routed parameter groups.
- Gradient clipping and optional dynamic LR normalization.
- Recurrence-aware gradient averaging by active steps.
- Adaptive loss-filter mechanism for outlier micro-batches.

## Kernel and model-path engineering

- Integrated Triton fused MLP path (`triton_mlp.py`).
- Maintained architecture-level toggles (smeargate, bigram hash, level signal, LoRA scope).
- Added quantization/export tooling for compact competition artifacts.

## Data diagnostics tooling

- Added dataset region analysis artifacts (e.g., `artifacts/region_scan_report.json`).
- Enabled better visibility into token distribution anomalies and outlier regions.
