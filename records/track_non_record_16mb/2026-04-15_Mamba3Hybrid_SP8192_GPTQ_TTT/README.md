# Mamba-3 Hybrid SSM + SP8192 + Legal TTT — 1.1473 bpb

Non-record submission. See PR description for full details.

## Run

```bash
# Install Mamba-3
bash setup_mamba3.sh

# Generate SP8192 data (~35 min)
cd data && python3 download_hf_docs_and_tokenize.py \
  --output-root . --tokenizer-config tokenizer_specs_8192.json --skip-byte

# Train + eval
VOCAB_SIZE=8192 NUM_LAYERS=7 NUM_ATTN_LAYERS=2 USE_BIGRAM_HASH=0 TRAIN_SEQ_LEN=4096 \
WARMDOWN_ITERS=2600 WARMDOWN_SHAPE=linear MUON_EQ_R=1 \
LATE_QAT_THRESHOLD=0.15 USE_GPTQ=1 QUANT_BITS=6 QUANT_BITS_EMBED=8 GPTQ_NUM_SEQS=32 \
EVAL_OVERLAP=1024 USE_LZMA=1 EVAL_TEMP=0.9 \
WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```
