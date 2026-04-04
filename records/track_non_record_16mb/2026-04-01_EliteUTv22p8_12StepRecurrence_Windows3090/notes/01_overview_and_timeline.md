## 01 — Overview and Timeline

This document is the top-level research narrative for this repository.

### Goal

Track how the project evolved from an early Universal Transformer recurrence setup into the current architecture/training system, under a hard 10-minute wallclock budget on Windows RTX 3090.

### Primary sources

- `logs/bpb_full_journey.csv` (main journey backbone)
- `logs/SWEEP_*.txt`, `logs/REFINE_*.txt` (UT-era tuning)
- `logs/AB2_*.txt`, `logs/AB3_*.txt` (later architecture sweeps)
- `results/ab3_sgbo_fixed/summary.csv` (AB3 feature summary)

## Timeline (BPB Journey)

| Phase | Milestone | Val BPB | Why it mattered |
|---|---|---:|---|
| Early UT | `Initial_UT_12step` | 3.75 | First stable recurrent baseline |
| Early UT | `Best_3090_12step` | 3.18 | Core UT stabilization (warmup/clip/safety) |
| Throughput Pivot | `dim512_discovery` | 2.7335 | More optimizer steps in wallclock beat slower depth strategy |
| New Regime | `mlp2_steps2_LR012` | **1.8085** | Major architecture shift with strong BPB drop |
| New Regime | `mlp5_steps1_lora512_BEST` | 1.8117 | Confirms new family strength in journey sheet |

## Interpretation

1. **Depth recurrence alone was not enough** under strict wallclock.
2. **Throughput-aware design changes** (faster steps, shape changes) produced the largest gains.
3. The codebase is now better described as an **architecture-search-evolved recurrent causal LM system**, not just “UT with depth recurrence.”

## Important validity note

`logs/bpb_full_journey.csv` contains one explicit invalid result:

- `SmearGate_BUG` with 0.7690 BPB is marked invalid due to non-causal leakage.

This must remain documented as invalid in all summaries.
