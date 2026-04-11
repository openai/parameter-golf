# TTT-LoRA 512d 8h 4kv warmdown-3000

## Summary

This submission explores a baseline-sized transformer (512d, 8 heads, 4 KV heads) with LoRA-based test-time training under the `track_10min_16mb` constraint.

The run was trained on 8×H100 SXM GPUs with a 10-minute wallclock cap. The final compressed artifact (int8 + zlib) stays within the 16 MB limit.

Key result: the best performance comes from the **TTT-LoRA evaluation path**, not the plain int8 roundtrip.

---

## Key Findings

- LoRA test-time training significantly improves final compression performance.
- The final score (`val_bpb=1.1957`) is achieved using `final_int8_ttt_lora`.
- Standard int8 quantization introduces a measurable gap (~0.036 BPB), which is partially recovered via LoRA adaptation.
- Staying within the 10-minute training budget is critical — training efficiency matters more than over-optimization.

---

## Results

| Metric | Value |
|------|------|
| Steps completed | 11,387 |
| Step average | ~52 ms |
| Total submission size (int8 + zlib) | 15,880,385 bytes |
| final_int8_zlib_roundtrip_exact val_bpb | 1.23159247 |
| final_int8_ttt_lora val_bpb | **1.1957** |

---

## Model Configuration

- **Vocab size:** 1024  
- **Model dimension:** 512  
- **Layers:** 9  
- **Attention heads:** 8  
- **KV heads:** 4  
- **Warmdown iterations:** 3000  
- **Max wallclock:** 600 seconds  

---

## Training Setup

- Hardware: **8×H100 SXM**
- Dataset: FineWeb (1024 tokenizer)
- Training stops automatically at wallclock limit (~600s)
- Final step reached: **11387**

---

## Quantization

- Final artifact uses:
  - **Per-row int8 quantization**
  - **zlib compression**
- Final size:
  - `15,880,385 bytes` (valid under 16MB)

---

## Evaluation Paths

### 1. Standard int8 roundtrip
- `val_bpb = 1.23159247`

### 2. TTT-LoRA (final score)
- `val_bpb = 1.1957` BEST

---

## Primary Training Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=baseline_8xh100_512_wd3000 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
