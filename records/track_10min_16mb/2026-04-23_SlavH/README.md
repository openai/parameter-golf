# Record: Nairi-Micro - Parameter Golf Submission

## Result: val_bpb = 0.9982 (Mean)


## Results

| Metric | Value | Limit |
|--------|-------|-------|
| **val_bpb (mean)** | **0.9982** | < 1.00 |
| **Artifact size** | **13.8 MB** | < 16 MB |
| **Training time** | **9 min 40 sec**| < 10 min |

## Results by Seeds

| Seed | val_bpb |
|------|---------|
| 42   | 0.9982  |
| 1337 | 0.9980  |
| 2025 | 0.9985  |
| **Mean** | **0.9982** |

## Architecture 
10L/576-dim Transformer. Increased capacity to the edge of the 16MB constraint.
## Quantization 
Mixed-Precision Int5/Int6 with QAT (Quantization-Aware Training). This allows storing more parameters while maintaining high precision for critical weights.
## Test-Time Training (TTT)
Implemented legal test-time adaptation, allowing the model to refine its weights based on the current context during evaluation.
## Optimizer
Muon (Newton-Schulz orthogonalization) combined with WSD (Warmup-Stable-Decay) scheduling for faster, more stable convergence within the 10-minute training limit.

## How to Run

```bash
# Set environment
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export VOCAB_SIZE=1024

# Execute Training
python train.py
```
