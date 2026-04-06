# Scylla + GPTQ + BH3072 — val_bpb 1.0856 (3-seed mean)

**val_bpb = 1.0856** (3-seed mean) | 15.3-15.8 MB | 8xH100 SXM | No SLOT, No TTT

## 3-Seed Results

| Seed | Sliding BPB | Artifact |
|------|------------|----------|
| 1337 | 1.1009 | 15,267,156 |
| 42 | **1.0782** | 15,813,568 |
| 2024 | **1.0777** | 15,807,116 |
| **Mean** | **1.0856** | |

Beats merged SOTA (1.1147, PR #1019) by 0.029 BPB (14x significance threshold).

## Key Techniques

- **Scylla tokenizer** (998-vocab TokenMonster, PR #1143 @simon-marcus): 37% fewer tokens per byte vs SentencePiece 1024
- **AR self-gen Full Hessian GPTQ** (int6, Cholesky error compensation): 64 self-generated sequences for calibration
- **BigramHash 3072x112**: matching #1019's configuration
- **Architecture**: 11L 512d 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL, VE128, XSA all 11 layers, QK-Gain 4.0, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997) + SWA, Late QAT, LZMA-9, FA3

## Compliance

- No SLOT (no eval-time delta optimization)
- No TTT (no eval-time weight updates)
- No n-gram cache
- No network calls
- Tokenizer byte accounting via validated metadata (candidate.meta.npz)
- All artifacts under 16MB, all training under 600s

## Reproduction

```bash
VOCAB_SIZE=998 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
DATA_PATH=./data/datasets/fineweb10B_scylla \
TOKENIZER_PATH=./candidate.vocab TOKENIZER_META_PATH=./candidate.meta.npz \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires Scylla-retokenized FineWeb shards (see anthonym21/fineweb10B-scylla on HuggingFace).

## Credits

- Scylla tokenizer: @simon-marcus (PR #1143)
- Training stack lineage: PR #175 (@anthony-maio), PR #1019 (@abaybektursun)
- GPTQ: PR #1019 (@abaybektursun)
- VRL: ResFormer (arXiv:2410.17897)
