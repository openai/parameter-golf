# 10L-512dim-scaled

## Approach
Scaled the baseline from 9 to 10 transformer layers while staying under the 16MB int8+zlib constraint. Trained for 1000 iterations on a single NVIDIA GB10 (DGX Spark).

## Changes from baseline
- `NUM_LAYERS=10` (default: 9)
- `ITERATIONS=1000` (default: varies)

## Results
- **val_bpb**: 2.0630 (int8+zlib roundtrip verified)
- **Submission size**: 15.7 MB (int8+zlib)
- **Training time**: 92s on 1×GB10
- **Parameters**: 18.9M

## Hardware
- NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory)
- Single GPU, no multi-GPU scaling

## Next steps
- 8×H100 full training run
- Mixed-precision QAT (int5/int6)
- Muon optimizer tuning
- BigramHash / SmearGate local context modules
