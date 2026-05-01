# SpotlightLFB: One-Site Late Feature Bank + Aux-Int6 Compression

**3-seed mean exact sliding-window val_bpb: 1.1493**  
Best seed: **1.1488**  
Track: `10min / 16MB`  
Hardware: `8xH100 SXM`, `~598s` train budget  
Style: `architectural speed + compression co-design`

## Result

| Seed | Steps | Raw exact val_bpb | Roundtrip exact val_bpb | Sliding exact val_bpb | Total bytes |
|------|------:|------------------:|------------------------:|----------------------:|------------:|
| 42 | 7428 | 1.1715 | 1.17220041 | 1.14878290 | 15,892,279 |
| 2025 | 7416 | 1.1722 | 1.17283139 | 1.14948399 | 15,922,399 |
| 1337 | 7431 | 1.1724 | 1.17310873 | 1.14970294 | 15,922,547 |
| **Mean** | **7425** | **1.1720** | **1.17271351** | **1.14932328** | |
| **Std** | | **0.00039** | **0.00038** | **0.00039** | |

This is a 10-minute-track submission built from a dense causal trunk. The goal was to make a branch that stays fast on real `8xH100` runs, compresses cleanly under the hard decimal `16,000,000`-byte cap, and still carries a distinct architectural idea rather than being another small optimizer retune.

Log note:

- `train_seed1337.log` is the untouched raw cloud log and includes an earlier appended run before the final legal run block.
- Metrics in `submission.json` are computed from the final completed run block in that raw log, while the file itself is preserved unedited.

## What Is New Here

The core idea is a **one-site late feature bank**.

Instead of spreading extra lexical refinement across many late layers, this submission concentrates that work at a single chosen late insertion site. The added path is small, cheap, and very targeted:

- previous-token embedding
- hashed bigram embedding
- boundary / chunk-start flag
- one late insertion site only: `LFB_LAYERS=6`

That makes the extra pathway behave more like a spotlight than a floodlight. The trunk does the heavy semantic work; the late feature bank only nudges the model once, late enough that the representations are already rich and stable.

## Why It Is Different

Most of the strongest public runs in the repo lean on one of two patterns:

- more late sites or bigger helper modules
- eval-time adaptation such as TTT

This submission deliberately does neither. The differentiator is a **concentrated late lexical steering path** paired with **compression-aware export design**.

- **Speed:** one site keeps the runtime overhead low enough to reach about `7.4k` steps in the `598s` wallclock budget.
- **Compression:** the extra parameters live mostly in small auxiliary tables and projections, which compress more predictably than another full-width transformer block.
- **Quality:** the branch keeps the raw-to-roundtrip gap small while still improving the final sliding-window score enough to justify the extra machinery.

## Architecture

Main trunk:

- `NUM_LAYERS=10`
- `MODEL_DIM=448`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3.0`
- `TIE_EMBEDDINGS=1`
- `XSA_LAST_N=4`

Late refiners:

- `VE_ENABLED=1`
- `VE_DIM=96`
- `VE_LAYERS=8,9`
- `LFB_ENABLED=1`
- `LFB_DIM=80`
- `LFB_LAYERS=6`
- `LFB_BIGRAM_VOCAB_SIZE=2048`
- `SKIP_REFINER_HIDDEN_DIM=24`

Training and deployment:

- `SWA_ENABLED=1`, `SWA_EVERY=50`
- `COMPRESSION_SHAPE_*` enabled
- `EXPORT_MODE=int8_lzma`
- only the small auxiliary embedding tables are pushed to int6 during export

## Compression Story

The second real contribution is that compression was treated as part of the architecture, not as an afterthought.

The final export path keeps the main trunk on the safer `int8+lzma` route, but selectively pushes only the small auxiliary embedding tables to int6:

- `bigram.embed.weight`
- `ve_shared.embed.weight`
- `lfb_shared.bigram_embed.weight`
- `lfb_shared.prev_embed.weight`

At the same time, the tiny projection and refiner matrices are explicitly forced through quantization so that the artifact stays legal without relying on large keep-float exceptions.

That gives a good speed / size / quality tradeoff:

- mean raw exact val_bpb: `1.1720`
- mean roundtrip exact val_bpb: `1.1727`
- mean sliding exact val_bpb: `1.1493`
- worst total size across seeds: `15,922,547` bytes

In other words, the branch stays inside the hard cap on all three cloud seeds without blowing up the exported score.

## Legality / Evaluation

This submission is intended to be rule-clean under the current public guidance:

- no TTT
- no eval-time optimizer steps
- no calibration on train data during export
- no future-token access in the late feature bank
- exact sliding-window evaluation is reported from the exported model

The late feature bank only consumes prefix-safe signals, and the exporter only reads the trained checkpoint.

## Reproduction

This folder is meant to run directly from inside `records/...`.

Default command:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional overrides:

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
LOG_DIR=/workspace/parameter-golf/logs \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`
- `train_seed42.log`
- `train_seed2025.log`
- `train_seed1337.log`
- `submission.json`
