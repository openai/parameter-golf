# BESE: Base-Efficient Subword Encoding for Parameter Golf

## Summary

A novel two-layer tokenizer that replaces SentencePiece with a structured 288-token vocabulary (40 base + 248 BPE merges). By encoding the 11 most frequent English letters as single tokens and grouping the remaining 15 into 4 context-disambiguated groups, BESE shrinks the INT6 embedding table by ~276KB compared to SP1024 (108KB vs 384KB). This freed budget enables depth recurrence, parallel residuals, and an n-gram prior for eval-time logit tilt within the 16MB artifact limit.

This is the first custom tokenizer submission to the Parameter Golf challenge.

**val_bpb: 1.1531** (mean of 3 runs; sliding window stride=64, INT6+LZMA roundtrip, n-gram tilt)

---

## Table of Contents

1. [BESE Tokenizer Design](#bese-tokenizer-design)
2. [BPB Calculation — Correctness Proof](#bpb-calculation--correctness-proof)
3. [Architecture](#architecture)
4. [Results](#results)
5. [Artifact Size Budget](#artifact-size-budget)
6. [Reproducing](#reproducing)

---

## BESE Tokenizer Design

### The Problem with Standard Tokenizers at 16MB

SentencePiece with 1024 tokens (SP1024) uses 1024 embedding vectors of dimension 512. In INT6 quantization, each vector costs 512 × 6/8 = 384 bytes, so the full embedding table is ~384KB. With tied embeddings (shared input/output), this is a fixed cost. At 16MB total artifact budget, ~384KB for embeddings leaves less room for transformer layers.

BESE reduces vocabulary to 288 tokens: 288 × 384 bytes = ~108KB. The ~276KB saving is enough to fund an extra transformer layer or deeper recurrence, which directly improves BPB.

The trade-off: BESE produces longer token sequences per byte of text (more tokens needed to represent the same text), so the model must predict more tokens. The question is whether the saved embedding bytes buy enough model capacity to overcome this.

### Layer 1: 40-Token Structured Base Alphabet

Every Unicode character is mapped to a sequence of base tokens. The design assigns the most efficient encodings (fewest tokens) to the most frequent characters.

#### Token ID Map

```
ID  Token          Bytes  Description
--  -----          -----  -----------
 0  PAD             0     Padding
 1  BOS             0     Beginning of sequence
 2  EOS             0     End of sequence
 3  UNK             0     Unknown

 4  'e'             1     Single-letter tokens: the 11 most frequent
 5  't'             1     English letters. Each letter = 1 token = 1 byte.
 6  'a'             1     These 11 letters cover ~72% of all letter
 7  'o'             1     occurrences in English text.
 8  'i'             1
 9  'n'             1
10  's'             1
11  'r'             1
12  'h'             1
13  'd'             1
14  'l'             1

15  GROUP_0         0     Group tokens: encode less-frequent letters.
16  GROUP_1         0     A group token by itself represents 0 bytes —
17  GROUP_2         0     it's incomplete until paired with a position
18  GROUP_3         0     token (see below).

19  POS_0           1     Position tokens: complete a grouped letter.
20  POS_1           1     The (group, position) pair maps to exactly
21  POS_2           1     one letter = 1 byte.
22  POS_3           1

23  SPACE           1     ' '
24  PERIOD          1     '.'
25  COMMA           1     ','
26  NEWLINE         1     '\n'
27  QUESTION        1     '?'
28  QUOTE           1     ' or " (all quote variants)
29  OTHER_PUNCT     1     Any other punctuation or non-ASCII byte

30  DIGIT_0         1     '0'
31  DIGIT_1         1     '1'
 ...                      ...
39  DIGIT_9         1     '9'
```

#### The Group System

The remaining 15 English letters are encoded using 2-token codes: a **group token** (0 bytes) followed by a **position token** (1 byte). The pair always totals 1 byte, matching the UTF-8 byte count of the original letter.

```
Group 15 (u,f,b,z):  15+19='u'  15+20='f'  15+21='b'  15+22='z'   (6.57% aggregate)
Group 16 (c,w,v,j):  16+19='c'  16+20='w'  16+21='v'  16+22='j'   (6.35% aggregate)
Group 17 (m,y,k,x):  17+19='m'  17+20='y'  17+21='k'  17+22='x'   (5.35% aggregate)
Group 18 (g,p,q):    18+19='g'  18+20='p'  18+22='q'               (4.00% aggregate)
```

**Design choices:**
- Letters that commonly co-occur in English bigrams (e.g., 'c' and 'k', 'm' and 'p') are placed in DIFFERENT groups. This gives BPE cleaner merge patterns — a common bigram like "ck" becomes `[GROUP_16, POS_0, GROUP_17, POS_3]`, and BPE can merge the full 4-token sequence into one token without ambiguity.
- Groups are ordered by aggregate frequency (most common group gets the lowest ID).
- Within each group, letters are ordered by individual frequency (most frequent letter = position 0).

#### Handling Non-ASCII and Multi-Byte UTF-8

For any character not in the encode table (accented letters, CJK, emoji, etc.):
1. Compute the UTF-8 byte length of the character
2. Emit that many `OTHER_PUNCT` tokens (ID 29, 1 byte each)

This guarantees the token count exactly equals the UTF-8 byte count for any non-ASCII character. For example, `'e'` (U+00E9, 2 UTF-8 bytes) → `[29, 29]` (2 tokens, 2 bytes).

**Critical correctness detail** in `_text_to_base_tokens()` (bese_fast_bpe.py line 55-71): even for characters that ARE in the encode table, the code checks that `utf8_len == mapped_bytes`. If a character like an uppercase accented letter maps to a base letter but has a different UTF-8 length, it falls back to `OTHER_PUNCT * utf8_len`. This prevents any byte count mismatch.

### Layer 2: 248 BPE Merges

Standard byte-pair encoding is trained on 50,000 FineWeb documents using the base token sequences. The 248 most frequent adjacent token pairs are merged, producing token IDs 40-287.

Each BPE merge token's byte count is the sum of its two constituent tokens' byte counts. This is computed recursively and stored in a lookup table (`bese_fast_bpe.py:_build_bpt()`):

```python
for pair, new_id in self.merges:
    merge_bpt[new_id] = merge_bpt[pair[0]] + merge_bpt[pair[1]]
```

**Total vocabulary: 40 base + 248 merges = 288 tokens.**

The BPE merge table is saved as a JSON file (`tokenizer.json`, 3.5KB) and loaded at training time.

---

## BPB Calculation — Correctness Proof

Since this submission uses a non-standard tokenizer, BPB correctness is critical. Here is a complete walkthrough of how bits-per-byte is calculated.

### The BPB Formula

```
BPB = (val_loss / ln(2)) × (total_tokens / total_bytes)
```

Where:
- `val_loss` = mean cross-entropy loss in nats per token
- `total_tokens` = number of tokens scored during evaluation
- `total_bytes` = number of UTF-8 bytes those tokens represent

This is the standard tokenizer-agnostic BPB formula used by all Parameter Golf submissions. The only tokenizer-specific part is computing `total_bytes`.

### How total_bytes Is Computed

Each token's byte contribution is looked up from a pre-built array `base_bytes_lut` (line 358 in train_gpt.py):

```python
token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
```

This array is built by `FastBESEBPETokenizer.build_luts_for_training()`:

```python
def build_luts_for_training(self, device=None):
    vs = self.vocab_size  # 288
    has_leading_space = np.zeros(vs, dtype=np.bool_)  # all False for BESE
    is_boundary = np.zeros(vs, dtype=np.bool_)
    is_boundary[PAD_ID] = True
    is_boundary[BOS_ID] = True
    is_boundary[EOS_ID] = True
    is_boundary[UNK_ID] = True
    return (
        torch.tensor(self._bpt.copy(), dtype=torch.int16, **kwargs),  # bytes per token
        torch.tensor(has_leading_space, dtype=torch.bool, **kwargs),  # always False
        torch.tensor(is_boundary, dtype=torch.bool, **kwargs),
    )
```

**Key point**: `has_leading_space` is all-False for BESE. In SentencePiece, some tokens include a leading space that adds an extra byte. BESE handles spaces explicitly (SPACE_ID = 23, 1 byte), so there are no hidden leading-space bytes. The `has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]` term in the BPB calculation always contributes 0 for BESE.

### Byte Count Invariant

**For any input text, the sum of `BYTES_PER_TOKEN[t]` over all tokens `t` in the BESE encoding equals `len(text.encode('utf-8'))`.**

Proof by cases:
1. **Single-letter tokens** (IDs 4-14): 1 token → 1 byte. Correct: each is a single ASCII letter (1 UTF-8 byte).
2. **Grouped letters** (group + position): 2 tokens → 0 + 1 = 1 byte. Correct: each grouped letter is a single ASCII letter (1 UTF-8 byte).
3. **Space, punctuation, digits** (IDs 23-39): 1 token → 1 byte. Correct: all are single ASCII characters (1 UTF-8 byte).
4. **Non-ASCII characters**: `utf8_len` copies of OTHER_PUNCT → `utf8_len` bytes. Correct by construction.
5. **Case handling**: `ch.lower()` maps to the encode table, but the code checks `utf8_len == mapped_bytes`. For ASCII upper-case letters (1 UTF-8 byte), the mapped bytes are also 1, so the check passes. For non-ASCII upper-case variants with different byte lengths, the check fails and falls back to OTHER_PUNCT (case 4).
6. **BPE merge tokens**: byte count = sum of constituents, which recurse to base tokens above. Correct by induction.

### Self-Test Verification

`bese_fast_bpe.py` includes a self-test (line 472-515) that verifies byte count invariance on diverse test strings including multi-byte UTF-8:

```python
for text in test_texts:
    enc = tok.encode(text)
    bpt = tok.get_bytes_per_token_lut()
    tb = int(sum(bpt[t] for t in enc))
    ub = len(text.encode("utf-8"))
    status = "OK" if tb == ub else "FAIL"
```

All tests pass with zero mismatches.

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Vocab size | 288 (40 base + 248 BPE merges) |
| Layers | 11 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP multiplier | 3x (hidden dim = 1536) |
| Activation | LeakyReLU(0.5)^2 |
| Depth recurrence | Layers 3-5, 3 forward loops, activated at 35% training progress |
| Parallel residuals | GPT-J style, start at layer 7 |
| Partial RoPE | 16 dims |
| LN scale | Enabled |
| Value embedding | Enabled (dim=128, layers 9,10) |
| Bigram hash | 2048 vocab, 128 dim |
| XSA | Last 4 layers |

### Optimizer

| Parameter | Value |
|-----------|-------|
| Matrix params | Parallel Muon (momentum=0.99, warmup from 0.92 over 1500 steps) |
| Embedding/scalar params | Adam (beta1=0.9, beta2=0.95) |
| Matrix LR | 0.022 |
| Muon weight decay | 0.095 |
| Adam weight decay | 0.095 |
| Gradient clip norm | 0.3 |
| QK gain init | 5.0 |

### Training Schedule

| Parameter | Value |
|-----------|-------|
| Wallclock cap | 600 seconds |
| Warmup | 20 steps (reset-style) |
| Warmdown | 5000 iters (time-proportional linear decay) |
| EMA decay | 0.9965 |
| SWA | Every 50 steps when LR scale < 0.2 |
| Late QAT | INT6 STE, enabled when LR scale < 0.15 |

### Key Optimizations

- **Compiled Newton-Schulz (NS5)**: `torch.compile(zeropower_via_newtonschulz5, dynamic=True)` fuses the 5-iteration BMM orthogonalization loop into a single CUDA kernel. Warmed up before the 600s clock starts to avoid compile overhead.
- **Batched EMA**: `torch._foreach_mul_` + `torch._foreach_add_` replaces a Python for-loop over all state dict entries with a single fused CUDA call per step.
- **N-gram tilt at eval**: Pre-computed bigram/trigram frequency table (292KB zlib-compressed) from the first training shard. Applied as additive logit bias (`+beta*0.5`) during sliding window eval. The table is bundled inside the INT6+LZMA artifact and extracted before eval.

---

## Results

Three independent training runs with different random seeds. All runs use the same pre-exported BESE shards and tokenizer.

| Metric | Run 1 (seed=1337) | Run 2 (seed=42) | Run 3 (seed=314) |
|--------|-------------------|-----------------|------------------|
| Training steps | 5417 | 5645 | 5645 |
| Raw val_bpb (post-EMA) | 1.1502 | 1.1478 | 1.1499 |
| INT6 roundtrip val_bpb | 1.1817 | 1.1803 | 1.1765 |
| **Sliding window val_bpb** | **1.1554** | **1.1539** | **1.1499** |
| Training time | 600s | 600s | 600s |
| Artifact size | 12.72 MB | 12.75 MB | 12.82 MB |

**Mean sliding window val_bpb (3 runs): 1.1531**

### BPB Trajectory (Run 1)

```
step  500: val_bpb 1.4692
step 1000: val_bpb 1.3679
step 1500: val_bpb 1.3274
step 2000: val_bpb 1.3074
step 2500: val_bpb 1.2887   ← recurrence activates at step 2561
step 3000: val_bpb 1.2476
step 3500: val_bpb 1.2257
step 4000: val_bpb 1.2068
step 4500: val_bpb 1.1875
step 5000: val_bpb 1.1656
step 5417: val_bpb 1.1502   ← 600s wallclock cap
```

---

## Artifact Size Budget

| Component | Bytes | MB |
|-----------|------:|---:|
| Model INT6+LZMA (includes bundled n-gram table) | 12,635,100 | 12.64 |
| Code (train_gpt.py) | 88,802 | 0.09 |
| **Total** | **12,723,902** | **12.72** |
| Limit | 16,000,000 | 16.00 |
| **Margin** | **3,276,098** | **3.28** |

The n-gram frequency table (298,802 bytes raw, zlib-compressed) is bundled inside the `final_model.int6.ptz` artifact under the key `"ngram"`. During eval, it is extracted to a temporary file and loaded by the `NgramTilt` class.

---

## Data Preparation

BESE cannot directly consume SP1024 shards. Data preparation runs before the timed 600s window (untimed, per challenge rules):

1. **Decode SP1024 shards** (80 shards, 8B tokens) back to text using SentencePiece (80 parallel workers)
2. **Quality filter** — remove documents with <50 words, low vocabulary richness (<25% unique words), or 3+ boilerplate markers
3. **Difficulty scoring** — composite of average word length, vocabulary richness, and sentence length
4. **Curriculum sort** — easy-to-hard ordering by difficulty score
5. **Train BPE** — 248 merges on 50K documents using fast indexed pair counting (O(N log N))
6. **Re-encode and export** — BESE+BPE tokenization of all filtered docs into binary shards (224 parallel workers)
7. **Build n-gram table** — scan first training shard for bigram/trigram frequencies (224 parallel workers)

Steps 2 and 3 are fused into step 1 (run inside the decode workers) to avoid single-threaded bottlenecks on 224-CPU pods.

---

## Reproducing

Full reproduction requires the upstream SP1024 FineWeb data and an 8xH100 pod.

```bash
# 1. Set up upstream data
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# 2. Clone BESE project (includes data prep orchestrator)
cd /workspace
git clone https://github.com/mrbese/parameter-golf-bese.git bese

# 3. Run full pipeline (data prep + training + eval)
cd /workspace/bese
python scripts/runpod_v5.py --num-gpus 8

# 4. Or with pre-existing shards (skip data prep):
python scripts/runpod_v5.py --skip-prep --num-gpus 8
```

The `train_gpt.py` in this submission folder can also be run standalone with pre-exported BESE shards:

```bash
cd records/track_10min_16mb/2026-04-16_BESE_NovelTokenizer_288vocab/
VOCAB_SIZE=288 NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3 \
NUM_HEADS=8 NUM_KV_HEADS=4 \
DEPTH_RECURRENCE_START=3 DEPTH_RECURRENCE_END=5 \
DEPTH_RECURRENCE_LOOPS=3 DEPTH_RECURRENCE_ACTIVATION_FRAC=0.35 \
PARALLEL_RESIDUAL_START=7 QK_GAIN_INIT=5.0 \
MATRIX_LR=0.022 MUON_WD=0.095 ADAM_WD=0.095 EMA_DECAY=0.9965 \
WARMDOWN_ITERS=5000 \
TOKENIZER_PATH=./tokenizer.json \
DATA_PATH=/workspace/bese_shards_v5/ \
NGRAM_TILT_ENABLED=1 NGRAM_TILT_MAX_N=3 \
NGRAM_PRIOR_PATH=/path/to/ngram_table_v5.bin \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files in This Submission

| File | Description |
|------|-------------|
| `train_gpt.py` | Self-contained training script (imports tokenizer from same directory) |
| `bese_constants.py` | 40-token base alphabet, encode/decode tables, bytes-per-token LUT |
| `bese_fast_bpe.py` | Fast BPE training (O(N log N)) and encoding, `FastBESEBPETokenizer` class |
| `tokenizer.json` | Pre-trained BESE+BPE tokenizer (248 merges, 288 total vocab) |
| `submission.json` | Submission metadata |
| `train_log_run1.txt` | Full training log (run 1, seed=1337) |
| `train_log_run2.txt` | Full training log (run 2, seed=42) |
| `train_log_run3.txt` | Full training log (run 3, seed=314) |
| `README.md` | This file |

## License

MIT
