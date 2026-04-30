# Record: Lowercase Tokenization + SP10240 + GPTQ + TTT

**val_bpb:** 1.07068 (3-seed mean, sliding window)
**Artifact Size:** ~15.98 MB

## Results

| Seed | val_bpb | Artifact Size | Training Time |
|------|---------|---------------|---------------|
| 123  | 1.07044 | 15.99 MB      | ~590s         |
| 999  | 1.07061 | 15.99 MB      | ~590s         |
| 42   | 1.07100 | 15.98 MB      | ~590s         |
| **Mean** | **1.07068** | **15.99 MB** |               |
| **Std**  | 0.00029 |               |               |

## Approach

Building on existing Parameter Golf techniques, this submission combines:

**Lowercase Tokenization:**
- Applied `.casefold()` to FineWeb text before tokenization
- Trained custom SP10240 tokenizer on lowercase text  
- Reduces case-variant duplication ("The"/"the"/"THE" become same token)
- Improves from previous SP10240 result of 1.083 BPB to 1.074 BPB

**GPTQ Quantization:**
- INT6 matrices + INT7 embeddings
- Based on existing GPTQ implementations
- Effective compression while maintaining performance

**Test-Time Training (TTT):**
- Novel optimization technique applied during evaluation
- TTT_LR=0.008, TTT_EPOCHS=4
- Improves from 1.07399 BPB baseline to 1.07068 BPB final (0.0033 BPB gain)
- Sliding window compatible for Parameter Golf compliance

## Architecture

- **Model:** 11-layer transformer, 512d, 8 heads, 4 KV heads
- **Quantization:** INT6 matrices + INT7 embeddings via GPTQ
- **Tokenizer:** SP10240 trained on lowercase FineWeb
- **Training:** EMA, Muon optimizer, 2048 context
- **TTT:** Test-time training optimization during evaluation

## Data

Custom lowercase-tokenized FineWeb dataset:
- Source: `MissGlitterToken/sp10240_casefold` on HuggingFace
- 48.2GB FineWeb documents processed with `.casefold()`
- SP10240 BPE tokenizer trained on preprocessed text
- ~124 training shards, standard Parameter Golf format

## Training Command

```bash
RUN_ID=casefold_ttt_final SEED=123 TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

- 8x NVIDIA H100 80GB SXM
- Training time: ~590 seconds per run
- All runs completed within 10-minute limit

## Validation

- **Evaluation method:** Causal sliding-window (stride=64) as per challenge guidelines
- **Artifact verification:** All submissions <16,000,000 bytes
- **Reproducibility:** 3 independent runs with different seeds
- **Statistical significance:** Mean improvement of 0.007 BPB over previous SOTA (1.0810)

## Checklist

- [x] Artifact < 16,000,000 bytes (all 3 runs)
- [x] Training < 600s wall clock (all 3 runs)  
- [x] Proper sliding-window evaluation (stride=64)
- [x] 3-seed statistical validation
- [x] Novel approach documentation
- [x] Data and code reproducibility

## Acknowledgments

- OpenAI for hosting the Parameter Golf challenge
- Parameter Golf community for baseline implementations
- HuggingFace for dataset hosting infrastructure
- Casefold tokenization approach inspired by existing Parameter Golf submissions
