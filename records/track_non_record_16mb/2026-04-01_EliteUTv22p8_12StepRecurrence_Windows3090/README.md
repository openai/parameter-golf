# Parameter Golf — Windows Memory Index (Elite v22.8)

Quick-reference index for all notes in this folder. Updated for the **Elite Universal Transformer v22.8**.

---

## Files

| File | Contents |
|---|---|
| [01_bugs_and_fixes.md](notes/01_bugs_and_fixes.md) | 21+ bugs found + exact fixes for Windows/RTX 3090 |
| [02_windows_setup.md](notes/02_windows_setup.md) | Step-by-step environment setup from scratch |
| [03_training_guide.md](notes/03_training_guide.md) | Every training command + all env variables |
| [04_wrapper_internals.md](notes/04_wrapper_internals.md) | How `train_gpt_windows.py` patches scripts at runtime |
| [05_custom_kernel.md](notes/05_custom_kernel.md) | Triton MLP Fused Megakernel Technical Details |
| [06_performance_tuning.md](notes/06_performance_tuning.md) | Performance tuning checklist + throughput notes |
| [07_final_architecture.md](notes/07_final_architecture.md) | The **Elite Universal Transformer (12.2M Unique)** |
| [08_muon_polar_express.md](notes/08_muon_polar_express.md) | Quintic Minimax (Degree-5) Stability Fix |
| [09_elite_standard_v20_run.md](notes/09_elite_standard_v20_run.md) | v20 run notes + iteration history |
| [10_vectorized_dataset_cleaning.md](notes/10_vectorized_dataset_cleaning.md) | Dataset cleaning pipeline notes |
| [11_elite_standard_v21_high_heat_run.md](notes/11_elite_standard_v21_high_heat_run.md) | History of the High-Heat Stability Pivot |
| [12_elite_transformer_implementation_summary.md](notes/12_elite_transformer_implementation_summary.md) | **Latest: v22.8 Implementation Summary** |

---

## TL;DR — What We Did (Elite v22.8 Pivot)

1. **Architecture**: Finalized the **12-step Recursive Elite UT**. Reuses a 1024-dim block with **Coordinate (Step) Embeddings** and LoRA.
2. **Normalization**: Moved to **Strict Pre-Normalization** (v22.8). All RMSNorms removed from the residual path to allow deep state accumulation.
3. **Stabilization**: Applied a **Universal Gradient Averaging (1/12)** and a **20-step Maturity Ramp** to prevent recursive explosion.
4. **Optimizers**: Switched to **Muon "Polar Express" (Degree-5)** at **0.009 LR** (Option B) for perfect monotonic convergence.
5. **Kernels**: Integrated **Fused Triton MLP** for $X \times W + LeakyReLU^2$, significantly boosting Windows throughput.
6. **Efficiency**: Saturating the RTX 3090 at **524,288 tokens/step** with **12.5GB VRAM** footprint.

---

## The 3-Line Cheat Sheet

```powershell
# 1. Setup the 'Elite' Environment (One-time)
.\setup_elite_env.bat

# 2. Launch the 10-Minute Stabilization Test (55 Steps)
.\limits_test_10m.bat

# 3. Scale for Final Championship Run
.\final_run_10m.bat
```

---

## Key Gotchas

- **Never run `train_gpt.py` directly on Windows** → will crash with Flash SDP error. Always use `train_gpt_windows.py` (via .bat).
- **Iteration Lock**: `train_gpt.py` is now environment-aware. Ensure `ITERATIONS` and `MAX_WALLCLOCK_SECONDS` are set in shell or .bat.
- **Polar Express steps**: Always set `MUON_BACKEND_STEPS=5` for degree-5 minimax stability on Ampere (RTX 3090).
- **VRAM Control**: Activation checkpointing is used across the recursive blocks to keep logic depth 12 within 12.5GB.

---

## Current Status v22.8

- [x] **Elite Universal Transformer** (UT v22.8) architecture locked.
- [x] **Strict Pre-Norm** (Residual-free) verified.
- [x] **Polar Express** (Degree-5) stability verified at 0.009 LR.
- [x] **Fused Triton MLP Kernel** verified.
- [x] **Environment Awareness** (Iteration controls) verified.
- [x] **3.29 BPB** at Step 60 achieved.
- [ ] Sub-1.x BPB leaderboard submission...
