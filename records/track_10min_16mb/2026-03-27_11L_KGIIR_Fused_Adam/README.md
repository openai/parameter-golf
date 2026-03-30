# KGIIR Trajectory Mixing

**Author:** Adam Jacuch  
**Base Architecture:** [Abay Bektursun](https://github.com/abaybektursun)  
**Validation BPB:** 1.11837  
**Throughput:** 88 ms / step (8xH100)

## Overview
This submission introduces **KGIIR trajectory mixing**, a lightweight gated causal temporal mixing modification built on top of the base architecture from **Abay Bektursun**. The goal of this change is to improve short-range dependency handling under the 16 MB and 600 s challenge constraints without materially increasing runtime.

In this run, adding the KGIIR mixer improved validation BPB from **1.11923** to **1.11837** at **88 ms / step**.

## What is KGIIR?
KGIIR is a **structured gated causal 4-tap temporal mixer** applied before Q/K/V projection.

Rather than relying only on fixed token shifts, the module computes a learned per-channel mixture of the current hidden state and several recent timesteps in a single fused expression:

$$
x^{\text{mixed}}_t = f_0 x_t + f_1 x_{t-1} + f_2 x_{t-2} + f_3 x_{t-3}
$$

where $f_0,\dots,f_3$ are learned per-channel coefficients derived from a gated parameterization.

**The module is not a recursive IIR in the strict DSP sense; it is a structured gated causal 4-tap temporal mixer.** The naming reflects the original design intuition, but the implemented mechanism in this submission is a finite-tap causal mixer over recent hidden states.

## Why use this in Parameter Golf?
In the 16 MB regime, local dependency handling needs to be cheap. This mixer provides a small, fusion-friendly mechanism for combining the current state with recent states before attention, with minimal implementation complexity and no custom kernel requirement.

The intent is not to replace attention, but to give the model a compact way to handle some short-range/local structure outside the main attention computation.

## Controlled Experiment
This run was performed as a controlled architectural modification to the Bektursun baseline.

* **Baseline BPB:** 1.11923
* **KGIIR-augmented BPB:** **1.11837**
* **Net improvement:** **-0.00086 BPB**

This is a **single-seed result**. It should be interpreted as an architectural record rather than a statistically established leaderboard flip.

## Technical Notes
* **Fusion-friendly implementation:** The temporal mixer is implemented as a single vectorized PyTorch expression over padded lagged states.
* **No custom kernel dependency:** The implementation is designed to remain simple and efficient within the challenge runtime budget.
* **Runtime:** The run maintained **88 ms / step** on 8xH100.

## Submission Scope and Limitations
This submission provides a single-seed verification only. While the BPB improvement is measurable, it does not establish a 3-run mean or clear the challenge’s stronger statistical bar for a definitive SOTA reversal.

Due to limited personal compute resources, I am submitting this as an **architectural record** documenting a promising lightweight temporal mixing primitive, along with final weights, training code, logs, and metadata for reproducibility.

## Reproduction Settings
```bash
# Exact HParams from Bektursun SOTA + KGIIR
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337
```

## Acknowledgment

This submission builds directly on the baseline architecture and training recipe from Abay Bektursun.
