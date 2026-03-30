# Static phrase-memory 1xH100 grant attempt

## What I tested

I wanted to know if small models benefit from hardcoded trigram lookup tables. The idea: a 16MB model probably can't learn every common 3-word phrase from scratch in 10 minutes of training, so maybe a static hash table that maps trigrams to learned embeddings would give it a shortcut.

Short answer: it doesn't help. The params are better spent on a bigger backbone.

## How I tested it

Everything ran on 1xH100 with the full FineWeb SP1024 dataset, 600s wallclock, sliding window eval, EMA, late QAT, and BigramHash. The only variable was whether TrigramHash was present and how large it was.

I ran 5 experiments total:

| Run                    | Params | MODEL_DIM | Trigrams             | val_bpb (int8 roundtrip) | Artifact size |
| ---------------------- | ------ | --------- | -------------------- | ------------------------ | ------------- |
| Full backbone control  | 25.3M  | 512       | None                 | **1.2791**               | 21.6MB ❌     |
| Slim control           | 20.6M  | 448       | None                 | **1.3007**               | 17.4MB ❌     |
| Slim + Trigram 2048×64 | ~19M   | 448       | 2048 buckets, 64 dim | **1.3029**               | 17.3MB ❌     |
| Slim + Trigram 4096×96 | ~21M   | 448       | 4096 buckets, 96 dim | **1.3040**               | 17.9MB ❌     |

The fair comparison is the bottom three rows — same slimmed backbone, only the trigram config changes. The slim control without trigrams (1.3007) beat both trigram variants (1.3029, 1.3040).

## Why I think trigrams didn't work here

Three reasons, roughly in order of how confident I am:

1. **Byte budget tradeoff.** The trigram table eats params that would've been more useful as backbone weights. At this scale every megabyte matters.
2. **Not enough training steps.** On 1xH100 I got ~3,600 steps. The trigram embedding layer might need more steps to learn useful hash-to-embedding mappings. On 8xH100 you'd get ~13K steps — worth retesting there.
3. **BigramHash already covers most local patterns.** The baseline already has an 8192×128 bigram table. Trigrams may be redundant on top of that.

This lines up with PR #486's one-line comment ("trigram ablation confirmed negative at small scale") but as far as I can tell nobody published the actual controlled comparison before.

## What didn't fit under 16MB

None of my runs fit. The best result (1.2791) came from a 25.3M param backbone that compressed to 21.6MB. Getting this under 16MB would need int6 QAT or a smaller MODEL_DIM, both of which I didn't have credits to test.

## What I'd do with more compute

1. Apply int6 QAT to the 25.3M backbone — if it compresses to <16MB without BPB collapse, that's immediately submittable
2. Run on 8xH100 to get competition-comparable numbers (~13K steps instead of 3.6K)
3. Retest trigrams with more training steps — the negative result might reverse when the model has time to learn the hash embeddings
4. Try XSA4 + EMA interactions on the compressed backbone — this is the current frontier recipe

## Reproduction

```bash
# Use train_gpt_phrase_memory_1xh100.py as train_gpt.py

# Full backbone control (best BPB but over 16MB)
SEED=1337 RUN_ID=phrase_control_static TRIGRAM_BUCKETS=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Slim control (fair comparison baseline)
SEED=1337 RUN_ID=slim_control_notrigram \
BIGRAM_BUCKETS=4096 BIGRAM_EMBED_DIM=64 MODEL_DIM=448 \
TRIGRAM_BUCKETS=0 EVAL_STRIDE=128 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Trigram ablation A (2048 buckets)
SEED=1337 RUN_ID=phrase_small_2048x64 \
BIGRAM_BUCKETS=4096 BIGRAM_EMBED_DIM=64 MODEL_DIM=448 \
TRIGRAM_BUCKETS=2048 TRIGRAM_EMBED_DIM=64 EVAL_STRIDE=128 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Trigram ablation B (4096 buckets)
SEED=1337 RUN_ID=phrase_main_4096x96 \
BIGRAM_BUCKETS=4096 BIGRAM_EMBED_DIM=64 MODEL_DIM=448 \
TRIGRAM_BUCKETS=4096 TRIGRAM_EMBED_DIM=96 EVAL_STRIDE=128 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
