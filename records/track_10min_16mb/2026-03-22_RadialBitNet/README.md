# Radial-BitNet 16MB Titan

This submission presents an experimental compressed language-model design for the Parameter Golf 16MB track.

The approach combines:
- BitNet-style ternary-weight linear projections,
- a custom positional scheme called **Radial Encoding**,
- a custom optimizer called **FRO (Fractal Resonant Optimization)**,
- compressed post-training export under the official artifact-size accounting rule.

This is a public experimental submission intended to demonstrate a non-standard architecture under the Parameter Golf constraints. The attached reported result was obtained from a development run on non-target hardware. No claim is made in this README that the reported score has already been reproduced under the official 8xH100 SXM record-track environment.

## Summary

The goal of this design is to push model capacity as far as possible under the official submission artifact limit by combining:
- ternary-style projection behavior for major linear layers,
- reduced learned overhead,
- tied embeddings,
- compressed final export,
- a training setup optimized for short wall-clock execution.

Rather than following a conventional FP16 baseline recipe, this submission explores a more aggressive compression-oriented design.

## Key Ideas

### 1. BitLinear Expansion
All major projections (`Q`, `K`, `V`, `O`, and MLP projections) use BitNet-style ternary-weight forward behavior. The purpose is to reduce effective storage pressure while preserving as much model width and depth as possible within the artifact budget.

### 2. Radial Encoding
Learned positional embeddings are removed. Instead, position-dependent geometric features are injected analytically through `RadialEncoding(8)`. This reduces learned parameter overhead while retaining explicit positional structure.

### 3. FRO Optimizer
`FRO` is a custom optimizer designed for short-horizon convergence under highly quantized weight dynamics. It replaces AdamW in this submission and is part of the experimental contribution.

## Configuration

- **Layers:** 12  
- **Model Dimension:** 384  
- **Attention Heads:** 6  
- **KV Heads:** 2  
- **Vocabulary Size:** 1024  
- **Approximate Parameter Count:** 15.6M  

## Artifact Accounting

The submission script performs a post-training artifact audit using:
- counted source-code bytes from `train_gpt.py`
- compressed exported model bytes
- a final decimal-byte check against the official `16,000,000` byte submission limit

The audit is performed after training and writes the compressed model artifact physically to disk before measuring its byte size.

## Evaluation

The script implements tokenizer-agnostic BPB evaluation over the official validation shard format used by the challenge. In record-track mode, the script is designed to fail explicitly if required tokenizer or dataset files are missing.

Mock or debug behavior is only enabled when explicitly requested through environment flags.

## Reproducibility Notes

`train_gpt.py` is designed to:
- support distributed execution,
- run with explicit record-track failure behavior when required assets are missing,
- produce a final post-training artifact audit,
- run final validation before reporting the final result.

## Development Status

The result currently attached to this submission comes from a development run on non-target hardware. This repository entry is intended as a serious experimental submission and as a candidate for further validation under the official challenge hardware setting.

## Files Included

This submission includes:
- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`

## Notes

This submission should be interpreted as an experimental compressed-model approach, not as a claim of already-verified record-track performance on 8xH100 SXM.
