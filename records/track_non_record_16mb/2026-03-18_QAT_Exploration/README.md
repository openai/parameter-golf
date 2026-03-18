# QAT + Architecture Exploration (Non-Record)

This is a non-record submission demonstrating quantization-aware training (QAT) applied to the baseline architecture.

## Approach

**Quantization-Aware Training:** We add simulated int8 per-row quantization to all `CastedLinear` layers during training using a straight-through estimator (STE). The model learns weights that are robust to int8 rounding, reducing the gap between pre-quantization and post-quantization val_bpb.

The 4-hour baseline shows this quantization gap grows from 0.0072 BPB (10-min run) to 0.0325 BPB (4-hour run), making QAT increasingly valuable as training quality improves.

**Implementation:** A `fake_quantize_per_row()` function simulates the exact quantization pipeline used in `quantize_state_dict_int8` -- per-row int8 with fp16 scales. The training flag is threaded through the model so evaluation uses unquantized weights.

## Status

Work in progress. Local MLX smoke tests on Apple Silicon are too resource-intensive for meaningful validation -- training runs saturate compute on consumer hardware, which is precisely why H100 access is needed. Pending H100 validation for leaderboard metrics.

## Planned Extensions

- SEQUENCE parameter sharing (Takase & Kiyono 2023) for more effective depth
- BitNet b1.58 ternary exploration for 5x parameter budget
- NorMuon optimizer, value embeddings, vocabulary tuning
