# Stock Baseline GPT 17M sp1024 — A40 10min

**Track:** non_record_16mb  
**GPU:** 1× A40  
**val_bpb (int8+zlib roundtrip):** 3.2686  
**Submission size:** 5,596,364 bytes (~5.3 MB)

## Summary

Unmodified stock `train_gpt.py` run on a single A40 GPU using the `sp_bpe_1024` tokenizer (vocab size = 1024). Serves as an A40 baseline reference point.

## Run Configuration

| Parameter | Value |
|---|---|
| VOCAB_SIZE | 1024 |
| ITERATIONS | 30 |
| TRAIN_BATCH_TOKENS | 2,097,152 |
| VAL_LOSS_EVERY | 10 |
| MAX_WALLCLOCK_SECONDS | 600 |
| Model params | 17,059,912 |
| GPU | 1× NVIDIA A40 |

## Training Command

```bash
export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  ITERATIONS=30 \
  TRAIN_BATCH_TOKENS=2097152 \
  VAL_LOSS_EVERY=10 \
  MAX_WALLCLOCK_SECONDS=600
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results

| Step | val_loss | val_bpb |
|---|---|---|
| 0/30 | 6.9357 | 4.1077 |
| 10/30 | 7.1298 | 4.2226 |
| 20/30 | 5.6383 | 3.3393 |
| 30/30 | 5.5269 | 3.2734 |
| **int8+zlib roundtrip** | **5.5190** | **3.2686** |

Exact: `val_loss=5.51896729 val_bpb=3.26864330`

## Artifact Sizes

- Serialized model (fp32): 67,224,983 bytes
- Model int8+zlib: 5,548,678 bytes
- Code: 47,686 bytes
- **Total int8+zlib submission: 5,596,364 bytes**
