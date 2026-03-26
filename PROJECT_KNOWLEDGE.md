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

- `tokenizer/bese_constants.py` — Shared alphabet constants (single source of truth)
- `tokenizer/bese_tokenizer.py` — Base 38-token encoder
- `tokenizer/bese_bpe_tokenizer.py` — Full BESE + BPE system (this is the main file)
- `scripts/train_bpe_jsonl.py` — Train BPE merges from JSONL text data
- `scripts/export_shards.py` — Export binary shards for train_gpt.py
- `scripts/runpod_all_in_one.py` — All-in-one RunPod script (decode, train BPE, export, train)
- `integration/train_gpt_bese.py` — Fork of train_gpt.py supporting BESE+BPE (.json) tokenizers
- `docs/integration_guide.md` — How to plug into parameter-golf repo
- `docs/EXPERIMENT_RESULTS.md` — RunPod 1xH100 experiment findings
- `docs/SUBMISSION.md` — Non-record PR checklist
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

## Experiment results (March 25, 2026)

Ran on RunPod 1xH100 SXM ($2.69/hr). Full details: `docs/EXPERIMENT_RESULTS.md`

- Decoded 10K FineWeb docs from SP shard, trained 250 BPE merges (vocab=288), exported BESE shards
- **Baseline (SP1024):** val_bpb = 1.3319 (1356 steps, 1B tokens, 10 shards)
- **BESE+BPE:** val_bpb = 3.9143 (1189 steps, 12.6M tokens, 2 shards)
- BESE result is data-starved (80x less training data than baseline) — not a fair comparison
- Tokenizer works correctly: 100% byte checks pass, train loss drops normally, model is smaller (12.9 vs 13.6 MB)
- Bottleneck: pure-Python BPE trainer/encoder too slow for full-scale data

## Infrastructure

- GitHub fork: https://github.com/mrbese/parameter-golf
- RunPod CLI (`runpodctl`) configured for account omerbese@gmail.com
- RunPod balance: ~$9.23 remaining
- SSH key: `~/.runpod/ssh/RunPod-Key-Go`

## v2 Improvements (March 25, 2026)

- `tokenizer/bese_fast_bpe.py` — Fast BPE training (~50x faster) and encoding using indexed linked-list approach with priority queue merging. Replaces the O(merges*tokens) pure-Python with O(tokens*log(tokens)).
- `scripts/runpod_v2.py` — All-in-one RunPod script that decodes ALL 10 shards (not just 1), uses fast BPE, supports configurable model architecture, and runs fair BESE vs baseline comparison.
- `integration/train_gpt_bese.py` now includes:
  - LeakyReLU(0.5)² activation (replaces plain relu², ~0.003 BPB improvement)
  - EMA (exponential moving average, decay=0.997, ~0.002 BPB improvement)
  - Defaults: 11 layers, 3x MLP (was 9 layers, 2x MLP)

## What's left to do

1. **Run fair comparison on RunPod** — use `scripts/runpod_v2.py` with all 10 shards
2. **Try wider models** — with BESE vocab=288 saving ~295KB, experiment with 576d or 640d width
3. **Add more competition techniques** — SmearGate, BigramHash, Int6 GPTQ-lite, XSA on last 4 layers
4. **Final 8xH100 run** for submission-quality score
5. Submit non-record PR — `docs/SUBMISSION.md`

## Origin story (for the write-up)

The approach was developed from first principles without ML background:
- T9 phone input → variable-length encoding → recognized as Huffman coding
- QWERTY typewriter jams → context-aware grouping → recognized as mutual information minimization  
- Bionic Reading → information density insight → dropped uppercase encoding
- Applied BPE on top of structured alphabet → solved sequence length problem

## Competition context

- Challenge runs March 18 – April 30, 2026
- Current SOTA: 1.1194 BPB (LeakyReLU² + TTT + Parallel Muon, 11L/512d)
- Baseline: 1.2244 BPB (9L/512d/2x MLP, SP1024)
- Top stack: 11L, 512d, 3x MLP LeakyReLU², U-Net skips, XSA last 4 layers, EMA+SWA, SmearGate+BigramHash, Int6 GPTQ-lite, Muon
- Nobody has changed the tokenizer yet — BESE would be a first
- Width > depth at this constraint point (ternary submission: 768d/10L competitive)
- All submissions are MIT licensed, public GitHub PRs
- Tokenizer changes get extra scrutiny from OpenAI reviewers
- Non-record track accepts novel/interesting approaches even if they don't beat SOTA
