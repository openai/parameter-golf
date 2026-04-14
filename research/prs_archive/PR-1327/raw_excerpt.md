# PR 1327 — BESE Tokenizer: Byte-Efficient Structured Encoding (287 vocab)

**Author:** mrbese
**Claimed BPB:** 1.1276 (1 seed — seed 1337; val_loss 1.5378; pre-quant 1.1462)
**Artifact size:** 15,337,953 bytes (~15.3 MB)
**Seeds:** 1337 only
**Track:** 10min_16mb
**Base PR:** 1019
**Steps:** 6,932 at 86.57 ms/step

## Files retrieved
- `records__track_10min_16mb__2026-04-04_BESE_Tokenizer_287_XSA11__README.md`
- `records__track_10min_16mb__2026-04-04_BESE_Tokenizer_287_XSA11__submission.json`
- `records__track_10min_16mb__2026-04-04_BESE_Tokenizer_287_XSA11__train_gpt.py`

## Claimed changes (from README, verbatim)

> Novel contribution: BESE tokenizer only. All training architecture, optimizer, quantization, and evaluation code from PR #1019.

> BESE replaces the standard SentencePiece-1024 tokenizer with a 287-token vocabulary built from a structured 38-token base alphabet: 8 single-letter tokens (e t a o i n s r — 8 most frequent English letters); 5 T9-style consonant groups (remaining consonants grouped by phone keypad layout); 4 positional tokens (encode char position within groups); special tokens (space, newline, digit prefix, uppercase prefix, BOS/EOS/PAD/UNK); 10 digit tokens; 249 BPE merges learned on FineWeb.

> Tokenization Efficiency: SentencePiece-1024 vocab 1024, tokens/byte 0.76, embedding params 524,288. BESE-287 vocab 287, tokens/byte 0.51, embedding params 146,944. BESE achieves 33% fewer tokens per byte; embedding table 72% smaller.

> Design Philosophy: T9 phone keyboards (grouping consonants by frequency and phonetic similarity); Huffman coding intuition; Bionic Reading.

> BPB formula is tokenizer-agnostic: val_bpb = cross_entropy_loss * tokens_per_byte. Validation set identical (FineWeb first-50K-document split, decoded from SP-1024 binary shards back to raw text, re-encoded with BESE). BPE merges trained only on training data.

> Architecture inherited from PR #1019: 11 layers, XSA all 11, GPTQ int6 + LZMA preset=9, Muon (WD=0.04), BigramHash 3072 x 112, EMA decay=0.997 / SWA every 50, VE layers 9-10 dim=128, TARGET_MB=15.9.

> Data preparation: ~50 minutes on 32-vCPU CPU pod (one-time cost, not counted in 10-minute training budget).
