# BESE Tokenizer — Novel Structured Encoding for Parameter Golf

**Based on:** [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (1.1147 BPB, @abaybektursun) — all training architecture, optimizer, quantization, and evaluation code from PR #1019. **Novel contribution: BESE tokenizer only.**

**val_bpb: 1.1276** (sliding window, stride=64, single seed) | **15.3 MB** | 8xH100 SXM, 600s

This submission introduces a novel tokenizer to the PR #1019 architecture: **BESE** (Byte-Efficient Structured Encoding), a 287-token vocabulary built from first principles that replaces SentencePiece-1024.

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 1337 | 6,932 | 86.6 | 1.1462 | **1.1276** | 15,337,953 |

## Novel Contribution: BESE Tokenizer

BESE replaces the standard SentencePiece-1024 tokenizer with a 287-token vocabulary built from a structured 38-token base alphabet:

- **8 single-letter tokens**: `e t a o i n s r` (the 8 most frequent English letters)
- **5 T9-style consonant groups**: remaining consonants grouped by phone keypad layout
- **4 positional tokens**: encode character position within groups
- **Special tokens**: space, newline, digit prefix, uppercase prefix, BOS/EOS/PAD/UNK
- **10 digit tokens**: 0-9
- **249 BPE merges**: learned on FineWeb data, fully absorbing group/position tokens

### Tokenization Efficiency

| Tokenizer | Vocab | Tokens/Byte | Embedding Params |
|-----------|-------|-------------|-----------------|
| SentencePiece-1024 | 1,024 | 0.76 | 524,288 |
| **BESE-287** | **287** | **0.51** | **146,944** |

BESE achieves **33% fewer tokens per byte**, meaning the model sees more content per context window. The embedding table is 72% smaller, freeing parameter budget for the transformer layers. The tokenizer is fully self-contained — no external SentencePiece dependency needed.

### Design Philosophy

The alphabet was designed independently from first principles, drawing on:
- **T9 phone keyboards**: grouping consonants by frequency and phonetic similarity
- **Huffman coding intuition**: most frequent letters get dedicated tokens
- **Bionic Reading**: the idea that partial information (first letters) is sufficient for pattern recognition

## The Story Behind BESE

This tokenizer was built by a founder/engineer with no formal ML research background, arriving at established concepts independently through everyday analogies:

- **T9 phone keyboards** inspired the consonant grouping — pressing one key to represent multiple related letters
- **QWERTY keyboard layout** informed the frequency-based letter selection — why do the most-used letters get prime real estate?
- **Bionic Reading** (the speed-reading technique that bolds first letters) suggested that partial character information could be enough for pattern recognition
- **Huffman coding** principles emerged naturally from thinking about efficiency — give the most common symbols the shortest codes

The result was a tokenizer design that, when validated against the literature, independently rediscovered mutual information minimization and hierarchical encoding — but arrived there from a completely different starting point than traditional NLP research.

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Tokenizer | BESE-287 (38 base + 249 BPE) | **Novel** |
| Layers | 11 | PR #1019 |
| XSA | All 11 layers | PR #1019 |
| Quantization | GPTQ int6 + LZMA preset=9 | PR #1019 |
| Optimizer | Muon (WD=0.04) | PR #1019 |
| BigramHash | 3072 x 112 | PR #1019 |
| EMA/SWA | decay=0.997 / every 50 steps | PR #1019 |
| Value Embeddings | Layers 9-10, dim=128 | PR #1019 |
| TARGET_MB | 15.9 (selective pruning) | PR #1019 |

## BPB Correctness (Tokenizer Change Verification)

Since this submission uses a custom tokenizer, we provide proof that `val_bpb` is correctly calculated:

1. **BPB formula is tokenizer-agnostic**: `val_bpb = cross_entropy_loss * tokens_per_byte`. The model predicts next tokens in BESE encoding, and the loss is scaled by the actual tokens-per-byte ratio of the BESE tokenizer on the validation set.

2. **Validation data is identical**: The validation set is the same FineWeb first-50K-document split. We decode the original SP-1024 binary shards back to raw text, then re-encode with BESE. No documents are added, removed, or reordered.

3. **No information leakage**: BPE merges are trained only on training data (50K documents from training shards). The validation set is never seen during tokenizer training.

4. **Byte-level consistency**: Every byte in the original text is preserved through the decode-reencode pipeline. The BESE tokenizer is lossless — `decode(encode(text)) == text` for all inputs.

5. **Cross-check**: The `final_int8_zlib_roundtrip_exact` line in the training log reports the same `val_bpb` as the sliding window evaluation, confirming internal consistency.

## Acknowledgments

The training architecture (model, optimizer, quantization, evaluation) is entirely from [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun. The only novel contribution in this submission is the BESE tokenizer and the data preparation pipeline. Full credit to the PR #1019 authors for the SOTA training stack.

## Data Preparation

BESE requires pre-tokenized training data. The preparation pipeline:
1. Decode original SentencePiece-1024 binary shards back to text
2. Train BPE merges on 50K documents from the training set
3. Re-encode all shards with the BESE tokenizer
4. Write new binary shards in the same format

Total prep time: ~50 minutes on a 32-vCPU CPU pod (one-time cost, not counted in the 10-minute training budget).
