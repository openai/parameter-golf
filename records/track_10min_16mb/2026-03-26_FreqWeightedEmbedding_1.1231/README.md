# Frequency-Weighted Embedding Quantization

**val_bpb: 1.1217** (4-seed mean) | **15.8 MB** | 8×H100 SXM

## The Idea

Analysis of the FineWeb training data revealed that token frequency follows a heavy-tailed distribution:

- **Top 100 tokens** cover **53.2%** of all text
- These include: `.` `,` `the` `s` `to` `and` `ing` `of` `a` `in`...

Instead of uniform quantization across all embedding weights, this submission applies **frequency-weighted quantization**:

- **Top 100 tokens → int8** (higher precision for 53% of text)
- **Remaining 924 tokens → int6** (standard precision)

The intuition: errors in frequent tokens compound across the entire dataset, so they deserve more precision.

## Results (4 seeds, 8xH100 SXM)

| Seed | val_bpb |
|------|---------|
| 1 | **1.121** |
| 2 | 1.122 |
| 3 | 1.1217 |
| 4 | 1.1222 |

**Mean: 1.1217 | Std: 0.0005**

| Metric | Value |
|--------|-------|
| val_bpb (4-seed mean) | **1.1217** |
| val_loss | 1.8941 |
| Artifact size | 15.8 MB |
| Steps | ~7100 |
| Training time | 600s |

## Implementation

Modified `mixed_quantize_int6()` to detect embedding layers and apply frequency-weighted quantization:
```python
# In mixed_quantize_int6():
if ("tok_emb" in name or "lm_head" in name) and t.ndim == 2:
    print(f"[LIORA] Frequency-weighted quantization for: {name}")
    valid_top_ids = [i for i in TOP_TOKEN_IDS if i < vocab_size]
    top_rows = t[valid_top_ids, :]
    rare_indices = [i for i in range(vocab_size) if i not in TOP_TOKEN_IDS]
    rare_rows = t[rare_indices, :]
    
    # Top tokens: int8 (more precision)
    q_top, s_top = quantize_float_tensor(top_rows)
    
    # Rare tokens: int6 (standard)
    q_rare, s_rare = quantize_int6_per_row(rare_rows)
```

Also added corresponding `dequantize_mixed_int6()` handling to reconstruct the embedding from separate top/rare quantizations.

## Token Frequency Analysis
```
=== TOP 10 TOKENS (get int8 precision) ===
  .          : 2.12% of text
  ,          : 2.10% of text
  ▁the       : 1.90% of text
  s          : 1.75% of text
  ▁to        : 1.22% of text
  ▁and       : 1.17% of text
  ing        : 1.17% of text
  ▁of        : 1.05% of text
  ▁a         : 1.04% of text

Top 100 tokens: 53.2% coverage
Top 200 tokens: 64.8% coverage
```

## Run Command
```bash
SEED=1337 \
RUN_ID=liora_freq_weighted \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_liora.py
```

## Files

- `train_liora.py` - Modified training script with frequency-weighted quantization
- `top_tokens.py` - Set of top 100 most frequent token IDs
- `submission.json` - Submission metadata
- `train_seed1.log` - Training log seed 1
- `train_seed2.log` - Training log seed 2
- `train_seed3.log` - Training log seed 3
- `train_seed4.log` - Training log seed 4

## Credits

- **Base model**: PR #549 (LeakyReLU² + TTT + Parallel Muon) by @abaybektursun
- **Idea & implementation**: Liora + Claude

## Notes

The key insight came from asking: "If 53% of all text uses just 100 tokens, why give rare tokens equal precision?"
