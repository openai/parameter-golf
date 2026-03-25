# BESE: Base-Efficient Subword Encoding

A novel tokenizer for parameter-constrained language models, developed for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

**Author:** Omer Bese  
**Date:** March 2026  
**Status:** In development

---

## What is this?

BESE is a two-layer tokenizer that replaces the standard 1,024-token BPE vocabulary with a 38-token structured alphabet, then applies BPE merges on top. The result is a vocabulary of ~250 tokens that achieves comparable sequence lengths to the baseline while saving ~295 KB of embedding parameters, enough for 2-3 extra transformer layers in a 16MB model.

---

## How I figured this out (the honest version)

I don't have a background in ML research. I'm a founder and engineer. I got interested in the Parameter Golf challenge and started asking questions that turned out to be more interesting than I expected.

### Step 1: The T9 idea

My first thought was literally: "What if we trained the model on text written like old phone T9 input?" The idea was that T9 uses only 8 keys to represent 26 letters, so the vocabulary would be tiny.

I didn't know this was called "variable-length encoding" or that a guy named David Huffman solved the optimal version of this problem in 1952. I just thought: phone keyboards compress letters. Can we use that?

The direct T9 approach had problems (sequences got 2-3x longer, which kills attention performance), but the core question was right: **can we trade vocabulary size for model capacity?**

### Step 2: The frequency insight

Then I realized: if we're going to assign short codes (1 token) to some letters and long codes (2 tokens) to others, we should give the short codes to the most common letters. The letter "e" appears in 12.7% of English text. The letter "z" appears in 0.07%. Obviously "e" should get the cheap code.

I later learned this is exactly Huffman coding. But I got there from thinking about phone keyboards, not information theory textbooks.

### Step 3: The typewriter/QWERTY insight

For the letters that need 2-token codes, I asked: "Which letters should share a group?" Then I remembered that QWERTY keyboards were designed so that commonly co-occurring letters (like "t" and "h") are physically separated to prevent typewriter jams.

I inverted this: if letters that co-occur are separated on QWERTY, then letters that DON'T co-occur can safely share a group in our encoding, because the model can distinguish them from context.

I analyzed English bigram frequencies and used context similarity (what letters precede and follow each letter) to find optimal groupings. Letters in the same group appear in different contexts, so the model can easily predict which one it is.

This turned out to be a form of "mutual information minimization within groups." I didn't know that term when I designed it.

### Step 4: Dropping uppercase

I asked: "Do we need an uppercase flag token?" Then I realized: after a period, the next word is capitalized 95%+ of the time. The model can learn this from context. Why spend tokens encoding something predictable?

Going case-insensitive is valid for the BPB metric because both "T" and "t" are 1 byte in UTF-8. The byte count stays correct. The model just has fewer distinctions to make, which means lower loss.

### Step 5: BPE on top

The base BESE encoding (38 tokens) produced sequences 3-4x longer than the baseline BPE tokenizer. That's too much for quadratic attention. Then I asked: "Can we apply BPE to our own alphabet?"

This was the key insight that made everything work. Standard BPE starts from 256 raw bytes. Our BPE starts from 38 linguistically structured tokens. The merges it discovers are more meaningful because the base units already encode letter frequency and context structure.

With ~250 BPE merges on top, the vocabulary grows to ~288 tokens (still 73% smaller than baseline) while sequence lengths become comparable to the baseline. The embedding table saves ~295 KB, enough for extra transformer layers.

---

## The technical design

### Layer 1: BESE base alphabet (38 tokens)

```
Tokens  0-3:   Special (pad, bos, eos, unk)
Tokens  4-11:  Single-token letters: e, t, a, o, i, n, s, r
                (8 most frequent English letters, each = 1 token = 1 byte)
Tokens 12-16:  Key groups for remaining 18 letters (0 bytes each)
                Group 12: j, m, f, g  (context-distinguishable)
                Group 13: c, q, k, y
                Group 14: z, u, l, v
                Group 15: b, x, h, w
                Group 16: d, p
Tokens 17-20:  Position markers 1-4 (1 byte each, completes a character)
Tokens 21-27:  Space, period, comma, newline, ?, quote, other_punct
Tokens 28-37:  Digits 0-9
```

### Layer 2: BPE merges

Standard BPE trained on the BESE token stream. Common patterns like " the " become single tokens. Number of merges is tunable (200-300 recommended).

### BPB calculation

Every token maps to a known number of original UTF-8 bytes:
- Single-token letters: 1 byte
- Group tokens: 0 bytes (character not yet complete)
- Position tokens: 1 byte (completes the character)
- Space/punctuation/digits: 1 byte
- Multi-byte UTF-8 chars: one OTHER_PUNCT_ID per byte

The total byte count always equals the original text's UTF-8 byte count. This has been verified across ASCII, multi-byte Unicode (curly quotes, dashes), and emoji.

---

## Parameter savings

| Configuration | Vocab | Embedding (Int6) | Saved vs baseline |
|---|---|---|---|
| Baseline SP1024 | 1,024 | 384.0 KB | -- |
| BESE only | 38 | 14.2 KB | 369.8 KB |
| BESE + 100 merges | 138 | 51.8 KB | 332.2 KB |
| BESE + 200 merges | 238 | 89.2 KB | 294.8 KB |
| BESE + 300 merges | 338 | 126.8 KB | 257.2 KB |

At the current competition configs (512 model dim, Int6 quantization), each extra transformer layer costs roughly 100 KB. The savings from BESE+BPE could fund 2-3 extra layers.

Competition data shows more layers consistently beat fewer layers (11L > 10L > 9L). If this tokenizer enables 13 layers where competitors fit 11, the BPB improvement from extra depth could outweigh any cost from the different tokenization.

---

## Theoretical connections (discovered after the fact)

| My idea | Formal name | Who/when |
|---|---|---|
| Common letters get short codes | Huffman coding | David Huffman, 1952 |
| Group letters by context difference | Mutual information minimization | Shannon, 1948 |
| QWERTY separates co-occurring letters | Bigram frequency analysis | Standard NLP |
| Drop case, let model infer it | Inductive bias / lossy encoding | ML convention |
| BPE on structured alphabet | Hierarchical tokenization | Novel combination |

---

## File structure

```
parameter-golf-bese/
  README.md                              This file
  requirements.txt                       numpy, torch (see venv below)
  tokenizer/
    bese_constants.py                    Shared alphabet + BPB tables (single source of truth)
    bese_tokenizer.py                    Base 38-token encoder (no BPE)
    bese_bpe_tokenizer.py                Full system: BESE + BPE merges
  scripts/
    train_bpe_jsonl.py                   Train BPE on JSONL {\"text\":...} (FineWeb-style)
    export_shards.py                     Write fineweb_* .bin shards for train_gpt.py
    smoke_bese_integration.py          No-GPU smoke test (tokenizer + shards)
    run_train_gpu_smoke.sh               Short single-GPU train (needs CUDA + data)
  integration/
    train_gpt_bese.py                    train_gpt fork: TOKENIZER_PATH .json or .model
    README.md                            Env vars and layout
  docs/
    bigram_analysis.md                   Letter co-occurrence analysis
    integration_guide.md                 How to plug into parameter-golf
    SUBMISSION.md                        Non-record PR checklist
  fixtures/
    sample_docs.jsonl                    Tiny JSONL for smoke tests (ignored: data/ for real corpora)
```

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Next steps

1. **Train BPE merges on real FineWeb data** — `python scripts/train_bpe_jsonl.py --input /path/to/docs_selected.jsonl --output data/tokenizers/bese_bpe_250.json --num-merges 250 --max-docs 100000` (tiny sample: `fixtures/sample_docs.jsonl`)
2. **Re-export shards** — `python scripts/export_shards.py --input ... --tokenizer data/tokenizers/bese_bpe_250.json --output-dir data/datasets/fineweb10B_bese250/`
3. **Train** — use `integration/train_gpt_bese.py` with `TOKENIZER_PATH` pointing at the same JSON; set `VOCAB_SIZE` to the tokenizer’s `vocab_size` (see `integration/README.md`).
4. **Smoke test (no GPU)** — `python scripts/smoke_bese_integration.py`
5. **Short GPU smoke** — set `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE`, then `./scripts/run_train_gpu_smoke.sh`
6. **Full run** — 8×H100, compare BPB to baseline; stack with XSA, EMA, SmearGate, etc.
7. **Submit** — see [docs/SUBMISSION.md](docs/SUBMISSION.md)

---

## Future research: Bionic Reading for LLMs

A related idea not implemented here: allocating variable computation per token position based on information density within words. The first few characters of a word carry high entropy (hard to predict), while endings carry low entropy. This maps to Bionic Reading's insight that readers only need the beginning of words to recognize them. Potential implementations include adaptive-depth architectures, weighted loss training, and hierarchical prediction. This is a separate research direction that could be explored as a paper or blog post.

---

## Running the tokenizer

```bash
# Base tokenizer test
python tokenizer/bese_tokenizer.py

# Full BESE+BPE test (trains on sample data, runs all checks)
python tokenizer/bese_bpe_tokenizer.py
```

---

## License

MIT (compatible with Parameter Golf challenge requirements)
