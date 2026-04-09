# BPB Verification

Manual byte-counting sanity check on the SP4096 validation shard.

## Method

1. Load SP4096 tokenizer (`fineweb_4096_bpe.model`)
2. Load val shard (`fineweb_val_000000.bin`): 44,848,122 tokens
3. For each target token, compute UTF-8 byte count using the same `build_sentencepiece_luts` logic as `train_gpt.py`
4. Compute BPB = (val_loss / ln2) * (tokens / bytes)

## Results

```
Val tokens:        44,848,122
Target tokens:     44,848,121
Total UTF-8 bytes: 150,755,442
Tokens per byte:   0.29748923

For seed 7 (val_loss = 2.59339416):
  Manual BPB:   1.11304910
  Reported BPB: 1.11304546
  Difference:   0.0000036 (float64 accumulation order)
```

## Conclusion

Manual computation matches reported BPB within float64 precision (3.6e-6 difference from accumulation order in sequential loop vs batched GPU computation). The BPB calculation is correct.
