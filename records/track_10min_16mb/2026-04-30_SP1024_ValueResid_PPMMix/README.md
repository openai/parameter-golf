# SP1024 + Value Residual + Byte-Level PPM Mixture

## Overview

This submission is the result of an incremental research process rather than a single clean-sheet design.

The training script was built step by step across many rounds of experiments. Instead of hard-coding one fixed model, we kept most architecture, optimization, tokenizer, and evaluation ideas behind environment-controlled switches so we could run controlled ablations quickly and compare many alternatives within one stable framework.

The final submission in this folder is a **record 16MB submission** based on:

- SentencePiece 1024 tokenizer
- 9-layer Transformer
- model dimension 512
- 8 attention heads / 4 KV heads
- MLP multiplier 2
- Value Residual enabled in the last 2 layers
- byte-level PPM mixture during final evaluation

## Submission Type

This is a **record submission**.

The included best run was produced on **1×H100**, with 600s as the wall clock. We do **not** claim verified compliance with the official **8×H100 / 10-minute** leaderboard requirement in this folder.

However, this run does satisfy the artifact-size requirement:

- compressed model: `15,650,103 bytes`
- code size: `156,032 bytes`
- total submission size: `15,806,135 bytes`

This fits under the 16MB limit.

## Best Included Result

### Neural roundtrip score
- `final_int8_zlib_roundtrip_exact val_bpb = 1.29339954`

### Final mixed score
- `ppm_mix_bpb = 0.829467`

This was the strongest included result for the SP1024 compact line.

## Main Idea

Our final direction is intentionally simple:

1. keep a compact Transformer backbone
2. improve the late value path with **Value Residual**
3. combine the neural model with a **byte-level PPM mixture** at evaluation time

In our experiments, this combination was more useful than continuing to add more complicated architectural branches.

## How the Code Evolved

This codebase was not written as a minimal one-off competition script.  
It evolved as a research scaffold.

Over time, we added switches for many ideas so that the same script could be reused for many sweeps and fair ablations. The broader script supports experimentation with:

- tokenizer variants
- BiFPN / BiFPN2 skip fusion
- XSA
- N-gram augmentation
- Value Residual
- cross-layer V and KV sharing
- PLE
- MTP
- parallel residual variants
- parallel-v2 side lanes
- LoRA-TTT
- byte-level PPM mixture

Many of these ideas were explored, but the strongest compact SP1024 line for this submission ended up being:

**compact backbone + value residual + byte-level mixture**

## Experimental Summary

A short summary of the findings that most influenced this submission:

### 1. Tokenizer choice mattered
Earlier sweeps showed that tokenizer choice had a large impact on compression performance. We explored SP1024, SP4096, and SP8192. For this submission, we chose SP1024 because it provided a compact, size-friendly line suitable for a 16MB submission.

### 2. Capacity still mattered
Increasing backbone capacity often helped, but for this submission we prioritized a compact model that still achieved a strong mixed score while fitting under the 16MB limit.

### 3. Value Residual was the strongest late-layer architectural improvement
Across many later Transformer ablations, **Value Residual** was the most consistent improvement that survived repeated testing. In this submission we enable it only in the last 2 layers.

### 4. Byte-level PPM mixture produced the largest final gain
The final score improvement came primarily from combining the neural model with a **byte-level PPM mixture** rather than from continuing to add more neural-only complexity.

## Final Configuration

Key settings for the included run:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `VALUE_RESIDUAL_ENABLED=1`
- `VALUE_RESIDUAL_LAST_N_LAYERS=2`
- `BIFPN2_MODE=1`
- `XSA_ENABLED=1`
- `NGRAM_MAX_N=2`
- `EMA_ENABLED=1`
- `LATE_QAT_RATIO=0.15`
- `PPM_ENABLED=1`
- `PPM_ORDER=5`
- `PPM_CONF_THRESHOLD=0.9`
- `LAMBDA_LO=0.10`
- `LAMBDA_HI=0.75`

## Included Files

This folder contains:

- `train_gpt.py` — final training and evaluation script
- `submission.json` — submission metadata
- `config.json` — selected configuration for the included run
- `requirements.txt` — Python dependencies
- `train.log` — log from the included best run
- `seed_runs.csv` — representative run summary

## Reproduction

A representative launch for the included run is equivalent to:

```bash
torchrun --nproc_per_node=1 train_gpt.py
