# 13L Int4-Packed MLP + Depth Recurrence + Pre-Quant TTT + Full Stack

**val_bpb: TBD** (pending evaluation on 8xH100)

## Novel Techniques

### True Int4 Bit-Packing (first in this competition)

Standard int4 quantization stores values in [-7,7] as full int8 bytes, wasting 4 bits per value. Our `pack_int4` function stores two int4 values in a single byte, cutting raw MLP storage in half before LZMA. Combined with Full Hessian GPTQ error compensation, int4 MLP achieves high reconstruction quality.

### 13 Layers (first submission beyond 11)

Int4 GPTQ + bit-packing saves ~3MB vs uniform int6, funding 2 extra transformer layers within 16MB. With depth recurrence on layers 4,5, the effective depth is 15 virtual layers.

### Pre-Quant TTT (adapted from top submissions)

After EMA weights are loaded and before GPTQ quantization, fine-tune the model with AdamW for 6 epochs (lr=0.0005, cosine decay, freeze first 2 blocks). This adapts the weights to the data distribution before quantization locks them in. Expected gain: -0.020 to -0.034 bpb.

## Run Command

```bash
# Default configuration (SP1024, 13L)
torchrun --nproc_per_node=8 train_gpt.py

# SP8192 mode (requires data: python3 data/cached_challenge_fineweb.py --variant sp8192)
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --nproc_per_node=8 train_gpt.py

# Disable TTT (faster eval, for testing)
TTT_EPOCHS=0 torchrun --nproc_per_node=8 train_gpt.py

# Try 14 layers (stretch)
NUM_LAYERS=14 XSA_LAST_N=14 VE_LAYERS=12,13 RECUR_LAYERS=5,6 torchrun --nproc_per_node=8 train_gpt.py
```

## Full Technique Stack

| Technique | Impact | Source |
|-----------|--------|--------|
| 13 layers + int4 packed MLP GPTQ | Novel: more depth in 16MB | Our innovation |
| True int4 bit-packing (pack_int4/unpack_int4) | Novel: 2 values/byte | Our innovation |
| Pre-Quant TTT (6 epoch AdamW) | ~-0.034 bpb | PR #1364, #1423 |
| Depth recurrence layers 4,5 | ~-0.005 bpb | PR #1204, #1420 |
| QK-Gain 5.0 | ~-0.005 bpb | PR #1217, #1423 |
| Trigram hash (zero extra params) | ~-0.002 bpb | Existing code, enabled |
| BigramHash 4096x112 | ~-0.001 bpb | Scaled up from 3072 |
| 3 VE layers (10,11,12) | ~-0.001 bpb | Extended from 2 |
| XSA all 13 layers | ~-0.005 bpb | PR #478, SOTA |
| LeakyReLU(0.5)^2 | ~-0.003 bpb | PR #493, SOTA |
| Full Hessian GPTQ (AR self-gen) | ~-0.007 bpb | SOTA |
| Parallel Muon + banks | Systems opt | SOTA |
| EMA(0.997) + SWA | ~-0.002 bpb | SOTA |
| LZMA preset 9 + selective pruning | Compression | SOTA |
| Sliding window eval (stride=64) | Eval opt | SOTA |

## Architecture

- 13 layers (15 virtual with recurrence), 512 dim, 8 heads, 4 KV heads (GQA)
- U-Net: encoder 6, decoder 7, 6 skip connections
- MLP 3x (hidden=1536), LeakyReLU(0.5)^2
- XSA all 13 layers, SmearGate, BigramHash(4096, 112), Trigram
- Value Embedding (dim=128) at layers 10, 11, 12
- Partial RoPE (16/64), LN Scale (1/sqrt(layer+1))
- Tied embeddings, logit softcap 30.0, QK-Gain 5.0

## SP8192 Migration Path

The current defaults use SP1024 (1024 BPE vocab). All top submissions use SP8192. To switch:
1. Download data: `python3 data/cached_challenge_fineweb.py --variant sp8192`
2. Set env vars: `DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192`
3. Expected additional gain: -0.015 to -0.020 bpb

## Configurable Parameters

All techniques are env var controllable for ablation testing:
- `NUM_LAYERS=13` (try 12 or 14)
- `RECUR_LAYERS=4,5` and `RECUR_EXTRA_LOOPS=1`
- `TTT_EPOCHS=6`, `TTT_LR=0.0005`, `TTT_FREEZE_BLOCKS=2`
- `QK_GAIN_INIT=5.0`
- `TRIGRAM=1`
- `BIGRAM_VOCAB_SIZE=4096`, `BIGRAM_DIM=112`
- `VE_LAYERS=10,11,12`

Built on SOTA by @abaybektursun and techniques from @clarkkev, @stukenov, @msisovic, @gowtham0992, @parinzee.
