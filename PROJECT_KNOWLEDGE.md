# BESE Project Context — Parameter Golf Challenge

## What this project is

We are building a novel tokenizer called BESE (Base-Efficient Subword Encoding) for OpenAI's Parameter Golf challenge (https://github.com/openai/parameter-golf). The challenge: train the best language model that fits in 16MB and trains in under 10 minutes on 8xH100s.

## The core idea

Instead of the standard 1,024-token BPE vocabulary (which wastes ~384KB on embeddings), we built a 38-token structured alphabet based on:

1. **Huffman-style frequency coding**: The 8 most common English letters (e,t,a,o,i,n,s,r) each get a single token. Less common letters get 2-token codes (group + position).

2. **Context-aware grouping (QWERTY-inspired)**: The 18 remaining letters are grouped into 5 groups based on bigram co-occurrence analysis. Letters in the same group appear in different contexts, so the model can disambiguate them easily. Groups: [j,m,f,g], [c,q,k,y], [z,u,l,v], [b,x,h,w], [d,p].

3. **Case-insensitive**: No uppercase flag. The model learns capitalization from context (after periods, proper nouns, etc.).

4. **BPE on top**: Standard BPE merges are applied to the BESE token stream, recovering sequence efficiency. With ~250 merges, total vocab is ~288 tokens, still saving ~295KB vs baseline while matching BPE sequence lengths.

## Key files on local machine

All code is at: `/Users/mrbese/Projects/parameter-golf-bese/`

- `tokenizer/bese_tokenizer.py` — Base 38-token encoder
- `tokenizer/bese_bpe_tokenizer.py` — Full BESE + BPE system (this is the main file)
- `docs/bigram_analysis.md` — Letter grouping optimization
- `docs/integration_guide.md` — How to plug into parameter-golf repo
- `docs/prior_art.md` — Literature review showing novelty
- `README.md` — Full project write-up with the discovery story

## Parameter savings

| Config | Vocab | Embedding (Int6) | Saved vs baseline |
|---|---|---|---|
| Baseline SP1024 | 1,024 | 384 KB | — |
| BESE + 250 merges | ~288 | ~108 KB | ~276 KB |

276KB savings ≈ 2-3 extra transformer layers at current competition configs.

## Vocabulary layout (38 base tokens)

```
0-3:    pad, bos, eos, unk
4-11:   e, t, a, o, i, n, s, r (single-token letters, 1 byte each)
12-16:  key groups (0 bytes each, incomplete character)
17-20:  position markers 1-4 (1 byte each, completes character)
21-27:  space, period, comma, newline, ?, quote, other_punct (1 byte each)
28-37:  digits 0-9 (1 byte each)
```

## BPB calculation

Critical for submission validity. Every token maps to a known number of UTF-8 bytes:
- Single-token letters = 1 byte
- Group tokens = 0 bytes
- Position tokens = 1 byte  
- Multi-byte UTF-8 chars = one OTHER_PUNCT_ID per byte
- Total always matches original text's UTF-8 byte count (verified and tested)

## What's left to do

1. Train BPE merges on real FineWeb data — `scripts/train_bpe_jsonl.py`
2. Re-export FineWeb shards — `scripts/export_shards.py`
3. Train with `integration/train_gpt_bese.py` (or patch upstream `train_gpt.py` per integration_guide.md)
4. GPU smoke — `scripts/run_train_gpu_smoke.sh`; full 8×H100 eval vs baseline
5. Combine with competition stack (XSA, EMA, SmearGate, etc.)
6. Submit non-record PR — `docs/SUBMISSION.md`

## Origin story (for the write-up)

The approach was developed from first principles without ML background:
- T9 phone input → variable-length encoding → recognized as Huffman coding
- QWERTY typewriter jams → context-aware grouping → recognized as mutual information minimization  
- Bionic Reading → information density insight → dropped uppercase encoding
- Applied BPE on top of structured alphabet → solved sequence length problem

## Competition context

- Challenge runs March 18 – April 30, 2026
- Current SOTA: ~1.12 BPB (non-TTT), ~1.07 BPB (with TTT)
- Baseline: 1.2244 BPB
- All submissions are MIT licensed, public GitHub PRs
- Tokenizer changes get extra scrutiny from OpenAI reviewers
- Non-record track accepts novel/interesting approaches even if they don't beat SOTA
