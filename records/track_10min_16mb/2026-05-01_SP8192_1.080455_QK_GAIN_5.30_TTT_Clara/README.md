# SP8192 QK-Gain 5.30 + Legal TTT, Python 3.11 H100 Run

This submission candidate is based on the public SP8192 record stack. It uses the SP8192 / 3-layer recurrence / parallel residuals / legal TTT lineage and changes the run configuration to `QK_GAIN_INIT=5.30`.

This is not a claim of a new architecture or an original method. Credit for the public 2026-04-09 SP8192 1.0810 record and the public records stack remains with the original authors. This entry documents a Python 3.11-compatible H100x8 run and wrapper packaging for the seed-42 result below.

## Result

| Seed | QK_GAIN_INIT | TTT LR | TTT epochs | Quantized sliding val_bpb | Quantized TTT val_bpb | Total submission size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.30 | 0.005 | 3 | 1.08181143 | 1.08045510 | 15,994,385 bytes |

Key metrics:

- `val_loss`: `2.79092773`
- `val_bpb`: `1.08045510`
- `quantized_val_loss`: `2.83742177`
- `quantized_val_bpb`: `1.09845439`
- `quantized_sliding_window_val_loss`: `2.79443127`
- `quantized_sliding_window_val_bpb`: `1.08181143`
- `quantized_ttt_val_loss`: `2.79092773`
- `quantized_ttt_val_bpb`: `1.08045510`
- Serialized model quantized+brotli: `15,977,752 bytes`
- Total submission size quantized+brotli: `15,994,385 bytes`
- Code size: `16,633 bytes`
- Eval end time: `Thu Apr 30 22:23:11 UTC 2026`

## Files

- `train_gpt.py` is the Python 3.11 wrapper used for the H100 run.
- `train_gpt_source.py` is the source script before wrapper generation.
- `make_py311_wrapper.py` is the wrapper generation helper from the H100 rawpack.
- `train.log` is copied from `run_seed42_qkgain530.log`.
- `full_log_3ef597c7.txt` is included as a background full-log backup.

## Run Command

The launch command was reconstructed from the run log and environment. The log records `seed=42`, `qk_gain_init=5.3`, `ttt_enabled=True`, `ttt_lr=0.005`, `ttt_epochs=3`, and `world_size=8`.

```bash
SEED=42 QK_GAIN_INIT=5.30 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Environment

- Hardware: H100 x8
- Python: 3.11
- PyTorch: 2.11.0+cu128
- CUDA: available
- `flash-attn-3`: installed and importable in the run environment

## Credits

This candidate builds on the public SP8192 record stack:

- `@bigbag` - public 2026-04-09 SP8192 1.0810 record / PR #1493 stack assembly
- `@clarkkev` - SP8192, GPTQ embeddings, SDClip, MuonEq-R, depth recurrence base work
- `@dexhunter` - 3-layer depth recurrence and legal TTT on SP8192
- `@abaybektursun` - score-first legal TTT framework and precedent
- `@Robby955` and `@msisovic` - parallel residuals lineage
- `@X-Abhishek-X` - public hyperparameter tuning lineage

No novelty beyond this run configuration, Python 3.11 compatibility, and wrapper packaging is claimed here.
