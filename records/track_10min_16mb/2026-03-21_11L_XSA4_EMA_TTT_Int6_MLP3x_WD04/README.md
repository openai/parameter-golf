# 11L XSA4 + EMA + Int6 MLP3x + Full-Model SGD TTT

**Mean val_bpb: 1.1442** (2 seeds on 8xH100 SXM), best: 1.1419 (seed 1337)

This is the highest-EV convergence branch. It keeps the strongest public training stack intact:
- 11 layers, 512 dim, 8 heads / 4 KV heads
- MLP 3x
- SmearGate + BigramHash(2048)
- OrthoInit + muP-style output scaling
- Muon/AdamW with WD=0.04
- int6 mixed quantization + zstd-22
- XSA on the last 4 layers
- EMA instead of SWA

Then it adds a single orthogonal eval-time change:
- full-model SGD TTT on the dequantized checkpoint
- 3 epochs
- lr=0.002
- momentum=0.9
- freeze the first 2 blocks

## Diverge, then converge

I considered four branches:
1. Keep the current best public training stack and add only TTT.
2. Keep the current best public stack and also retune batch size / RoPE / other knobs.
3. Go back to a 10L int5 / late-QAT branch.
4. Keep exploring byte-aware auxiliaries and curricula.

I converge on branch 1.

Reason:
- The 11L EMA + XSA stack is already the strongest public training-time base.
- Full-model SGD TTT is the strongest proven orthogonal eval-time add-on.
- The “stack many more things at once” branch already looks weaker and harder to reason about.
- The byte-aware branch already underperformed badly in practice.

## Why this differs from prior winners

Compared with the main winning branches:
- vs PR #198: this keeps the 11L SmearGate/Bigram/WD/int6 recipe but upgrades SWA to EMA and adds XSA.
- vs PR #287: this keeps the exact best public training stack and adds full-model SGD TTT.
- vs PR #254: this uses a much stronger base model before TTT.
- vs PR #290: this avoids batch-size and RoPE retunes and makes only one new move on top of the best base.
- vs my earlier byte-aligned run: this does not spend training budget on auxiliary objectives that failed to pay back.

## Practical notes

This script includes a FlashAttention-3 import fallback:
- if `flash_attn_interface` is available, it uses FA3
- otherwise it falls back to PyTorch SDPA and still runs

That makes it safer on the current RunPod template.

The artifact target still depends on `zstandard` being available. If the script falls back to zlib, quality may stay fine but the compressed artifact may no longer be competitive.

## Default config

The script defaults already encode the intended record-track settings:
- `NUM_LAYERS=11`
- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_SEQ_LEN=2048`
- `EVAL_STRIDE=64`
- `BIGRAM_VOCAB_SIZE=2048`
- `XSA_LAST_N=4`
- `EMA_ENABLED=1`
- `SWA_ENABLED=0`
- `MUON_WD=0.04`
- `ADAM_WD=0.04`
- `MATRIX_LR=0.025`
- `SCALAR_LR=0.025`
- `TIED_EMBED_LR=0.035`
- `TTT_ENABLED=1`
- `TTT_LR=0.002`
- `TTT_EPOCHS=3`
- `TTT_FREEZE_BLOCKS=2`

## Run

From repo root on an 8xH100 pod:

```bash
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_TTT_Int6_MLP3x_WD04/train_gpt.py
```

I would run three seeds first:

```bash
for SEED in 1337 42 2025; do
  RUN_ID=xsa4_ema_ttt_$SEED \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_TTT_Int6_MLP3x_WD04/train_gpt.py
done
```

## Results

| Seed | Pre-quant val_bpb | Post-int6 val_bpb | Post-TTT val_bpb | Steps | ms/step | Artifact bytes |
|------|-------------------|-------------------|-------------------|-------|---------|----------------|
| 1337 | 1.1581 | 1.1655 | **1.1419** | 5,344 | 109.2 | 15,578,775 |
| 1338 | 1.1616 | 1.1701 | **1.1464** | 4,559 | 131.6 | 15,661,221 |
| **Mean** | | | **1.1442** | | | |

Hardware: 8xH100 SXM (community cloud). SDPA fallback (no FA3).
Seed 1337: ~109ms/step. Seed 1338: ~132ms/step (different node).
TTT: 3 epochs SGD. Eval: stride-64 sliding window.
All artifacts under 16MB (zstd-22 compression).

## vs. Prior SOTA

| Run | val_bpb |
|-----|---------|
| Compression-Funded MLP3x (best seed) | 1.1598 |
| Compression-Funded MLP3x (mean) | 1.1647 |
| **This run (best seed 1337)** | **1.1419** |
| **This run (2-seed mean)** | **1.1442** |
| Delta (best vs best) | **-0.0179** |
| Delta (mean vs mean) | **-0.0205** |

## Compatibility fixes applied

- SDPA GQA fallback: manual KV head repeat for PyTorch <2.5 (no `enable_gqa`)
- RoPE cache clear before TTT: prevents "inference tensors cannot be saved for backward" error
- Requires `zstandard` pip package for zstd-22 compression (falls back to zlib otherwise, overshoots 16MB)
