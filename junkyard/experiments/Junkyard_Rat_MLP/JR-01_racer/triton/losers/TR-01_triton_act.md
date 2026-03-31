# TR-01 — Triton Activation Kernel Loser

Date: 2026-03-29

## Result

First kernel-backed Triton attempt on top of the `JR-01` winner.

| Variant | Step avg | Post-EMA BPB | Sliding BPB |
|---|---:|---:|---:|
| `JR-01` eager MLP | `91.00ms` | `1.1340` | `1.11056240` |
| `TR-01` `triton_act` | `91.11ms` | `1.1345` | `1.11099954` |

## Verdict

`TR-01` loses.

Reason:
- no speed win
- slight quality regression

## Why It Still Matters

This does **not** kill the Triton track.

It proves:
- the kernel path is stable in the real training loop
- Triton can be wired into this stack without blowing up training

So the next work is tuning and broader fusion, not pretending `TR-01` already won.

## Re-run

```bash
bash experiments/Junkyard_Rat/triton/run_jr02_triton_act.sh
```
