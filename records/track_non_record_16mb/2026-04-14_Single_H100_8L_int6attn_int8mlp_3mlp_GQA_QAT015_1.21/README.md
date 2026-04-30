# Go Small or Go Home

The goal of this submission is to optimize on a pure neural architecture on a single H100 GPU. ( so we shrink the model instead of increasing the size )

## Non-Record: 1.2192 BPB — 8L GQA + SP4096 + MLP3 + Partial RoPE + XSA + Mixed Quant + LR optimization + FlashAttention-3

**val_bpb = 1.2192** (3-seed mean sliding) | **~15.92 MB** max artifact | Single-H100

## 3-Seed Results

| Seed | Sliding BPB (s96) | Pre-Quant BPB | RT BPB | Artifact |
|------|-------------------|---------------|--------|----------|
| 1    | 1.2187            | 1.2325        | 1.2361 | 15,913,674 bytes |
| 42   | **1.2180**        | **1.2322**    | **1.2355** | 15,921,446 bytes |
| 1337 | 1.2209            | 1.2345        | 1.2380 | 15,910,345 bytes |
| **Mean** | **1.2192**    | **1.2331**    | **1.2365** | **15,915,155 bytes** |

## Architecture

- **SP4096 Tokenizer** instead of the default BPE vocabulary
- **8 Transformer Layers** (dim 512, reducing depth to prioritize parameters elsewhere)
- **GQA (Grouped Query Attention)** (8 query heads / 4 KV heads) to maximize parameter efficiency.
- **Partial RoPE (32 dimensions)**: applies Rotary Embeddings on only a subset of head dimensions, improving processing speed and flexibility.
- **XSA (Cross-Sequence Attention)**: Activated specifically on the deepest 2 layers (`xsa_last_n=2`).
- **MLP 3x width expansion**: tuned the multi-layer perceptron hidden size multiplier to 3x, yielding better parameter density.
- **Tied Embeddings**: shares weights between the token embedding and LM head.
- **FlashAttention-3** interface.
- **Initialization Upgrades**: Orthogonal variance scaling (gain=1.0) for large linear weights and projection scaling scaled by `1/sqrt(2 * num_layers)`.
- **RMSNorm Enhancements**: explicitly applying an epsilon of `1e-6`.

## Training & Techniques Changed from Base (`train_gpt.py`)

- **Muon & AdamW Optimizers with Direct Weight Decay**: Integrated direct weight decay scaling into the custom `Muon` Newton-Schulz algorithm step (`p.data.mul_(1.0 - lr * weight_decay)`). Transitioned the embedding optimizer to fused `AdamW`.
- **Wallclock-Aware Warmdown**: Dynamically tracks elapsed compute time to trigger a linear learning-rate decay over the final 20% of the 600-second compute budget.
- **Late QAT (Quantization-Aware Training)**: Over the final 15% of the run, a custom Fake-Quantization Straight-Through Estimator (STE) intercepts `CastedLinear` forward passes. 
- **Mixed INT6 and INT8 Storage (`mixed_int6_int8_per_row_v1`)**: Implemented bitwise packing (`pack_lowbit_tensor`) to shrink specific attention/MLP layers (`INT6_NAME_PATTERNS`) tightly into 6-bit payloads alongside INT8 formats.
- **Zlib Payload Compression**: The exported raw bytes are aggressively compressed with maximum zlib (`level=9`) upon export.
- **Sliding-Window Evaluation (`eval_val_sliding`)**: Validates strictly over a continuous slide (stride 96), isolating the scoring to properly context-informed tokens avoiding chunked boundary penalties.

## Reproduction

To setup the dataset and the sp4096 tokenizer, run the following:

```bash
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

Then, you can reproduce the results using:

```bash
RUN_ID=baseline_sp4096 \
torchrun records/track_non_record_16mb/2026-04-11_Single_H100_8L_int6attn_int8mlp_3mlp_GQA_QAT015_1.21/train_gpt.py
```
*(Note: `DATA_PATH`, `TOKENIZER_PATH`, and `VOCAB_SIZE` environments for SP4096 are now set as defaults inside the script).*

## Compliance

- [x] 3 seeds evaluated.
- [x] Max total submission artifact size <= 16,000,000 bytes (max 15.92MB).
- [x] No test-time training on validation data.
- [x] No network calls during evaluation.
- [x] No external compute.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `seed1.log`
- `seed42.log`
- `seed1337.log`

## Credits

- **@adityasasidhar** — Author
- **@clarkkev (Kevin Clark)** — SP4096 tokenizer scaling and high weight-decay integration logic.
- **@jfprincz** — Partial RoPE parameterization and initial XSA (Cross-Sequence Attention) implementations.
- **@abaybektursun & @unnir** — Explorations and efficiency optimizations around XSA setups.
- **@aruniyer & @raahilshah (Raahil Shah)** — MLP 3x expansion and early Int6 QAT structures.
- **@signalrush** — Late Quantization-Aware Training (QAT) design and warmdown scheduling.
- **@aquariouseworkman, @Nan Liu, & @thwu1** — Mixed precision quantizations (int6/int5/int8 blends) and initial sliding window concepts.
- **@Matthew Li** — Original formulation of the sliding window evaluation bridging scheme.
- **@dexhunter** — Muon optimizers and aggressive low-bit precision setups.
- **@MatoTeziTanka** — Thanks for your cool updates on discord 