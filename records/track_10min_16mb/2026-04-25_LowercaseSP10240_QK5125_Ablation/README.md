# Lowercase SP10240 QK Gain 5.125 Ablation

This is a controlled one-seed ablation, not a SOTA claim. It keeps the lowercase SP10240 tokenizer, depth recurrence, parallel residuals, Muon training setup, GPTQ INT6 matrices, INT7 embeddings, and Brotli compression, then changes the default QK gain from 5.0 to 5.125.

## Result

| Seed | Sliding BPB | Artifact bytes | Train time |
|------|-------------|----------------|------------|
| 42 | 1.07419003 | 15,985,150 | 588.029s |

Modal run: `ap-p5ABFuwbpOrepAmodKYjNj`

## Configuration

- Hardware: 8x NVIDIA H100 80GB HBM3 on Modal, requested with exact `H100!:8`
- Python: 3.12.1
- PyTorch: 2.9.1+cu128
- Dataset: `MissGlitterToken/sp10240_casefold`
- Tokenizer: lowercase SP10240 SentencePiece model
- `QK_GAIN_INIT=5.125`
- `TTT_ENABLED=0`
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=4000`

## Validation

- Training stopped by the wallclock cap at step 4528 with `train_time: 588029ms`.
- Post-EMA pre-quantized validation: `val_bpb=1.07560344`.
- Quantized standard validation: `val_bpb=1.09004268`.
- Quantized sliding-window validation: `val_bpb=1.07419003`, `eval_time=525027ms`.
- Total compressed artifact size was `15,985,150` bytes, under the 16,000,000 byte limit.
- No test-time training or eval-time adaptation was enabled.

## Reproduction

```bash
DATA_DIR=/path/to/data \
RUN_ID=lowercase_sp10240_qk5125_seed42 \
SEED=42 \
QK_GAIN_INIT=5.125 \
TTT_ENABLED=0 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=4000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

Compared with the matching seed-42 QK gain 5.0 baseline (`1.07389857`), this QK gain setting is slightly worse by about `0.00029` BPB. The run is still technically sound as a negative ablation because it completed the full quantized sliding-window evaluation, stayed within the artifact size cap, and used no eval-time adaptation.

## Credits

This ablation was run by `suryavanshi` on Modal.
