# Local Optimized 9L 2xMLP Parameter Sweep Configuration

This submission record was created following extensive analysis of the existing Modded-NanoGPT and Parameter Golf SOTA records and adapted to be debuggable on an 8GB VRAM graphics card locally (by increasing gradient accumulation inherently and avoiding sweeping the whole dataset on step 0), before scaling.

## Architecture

- **Layers**: `9`
- **MLP Multiplier**: `2`
- **Train Sequences**: `2048`
- **Warmdown Iterations**: `3600`
- **Matrix Learning Rate**: `0.055`

This model runs well within the parameter size limits while using an expanded architecture format favored by previous winning submissions.

To test this configuration without OOMing the local 8GB GPU memory, the script should be run with:

```bash
$env:TRAIN_BATCH_TOKENS="16384"
$env:VAL_BATCH_SIZE="16384"
$env:LOCAL_SMOKE_TEST="1"
python train_gpt.py
```
