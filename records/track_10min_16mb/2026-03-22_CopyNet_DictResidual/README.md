# Compression-Inspired Language Modeling: Copy Network + Dictionary Residual

**val_bpb: TBD** (pending 8xH100 verification)

## Key Idea

Classical compression algorithms like LZ77 and Brotli solve the same fundamental problem as language models: predicting what comes next in a sequence. We directly transplant two core compression principles into the neural architecture:

1. **LZ77 backreferencing → Copy Network**: When text repeats something that appeared earlier (names, terms, phrases), don't re-predict it from scratch — point back and copy.

2. **Brotli's pre-built dictionary → Dictionary Residual**: Common patterns (frequent bigrams/trigrams) are captured by a frozen statistical lookup table. The neural network only needs to learn what the dictionary can't predict.

## Architecture

```
Input tokens
     |
Token Embedding (dim=640, tied)
     |
Frozen N-gram Dictionary --> dict_logits (bigram probabilities, no gradient)
     |
Transformer (11 layers, 10 heads, 5 KV heads, 2x MLP)
     |
correction_logits <-- Neural correction head
     |
dict_weight = sigmoid(learned_scalar)
gen_probs = softmax(dict_weight * dict_logits + (1 - dict_weight) * correction_logits)
     |
Copy Attention --> copy_probs (pointer distribution over previous tokens)
     |
copy_gate = sigmoid(learned_gate(hidden))
     |
final_probs = (1 - copy_gate) * gen_probs + copy_gate * copy_probs
     |
NLL Loss
```

### Copy Network (LZ77-inspired)

A pointer mechanism that attends over all previous hidden states and produces a probability distribution over the vocabulary based on which previous tokens had similar contexts. Web text is highly repetitive -- names, technical terms, boilerplate phrases all recur within documents. The copy mechanism handles these for ~410K extra parameters (2.4% overhead).

The copy attention computes:
- Query: linear projection of current hidden state
- Keys: all previous hidden states
- Attention weights scattered to vocabulary via input token IDs
- A learned gate blends copy vs generate probabilities per token

### Dictionary Residual (Brotli-inspired)

A frozen bigram frequency table (computed from training data at startup) provides base predictions. The neural transformer only needs to correct what the dictionary gets wrong. This is directly inspired by Brotli compression, which uses a pre-built dictionary of common web strings and only compresses what the dictionary can't handle.

The dictionary:
- Built from the first 5M tokens of training data (unigram + bigram + trigram hash table)
- Registered as non-persistent buffers (not saved in the artifact, recomputed at load time)
- Zero parameters, zero gradient -- completely frozen during training
- A single learned scalar controls the dictionary-vs-neural mixing weight

### Why This Works

The two mechanisms are complementary:
- **Dictionary** handles predictable patterns: common word sequences, grammar, frequent phrases
- **Copy** handles repetitive patterns: names, terms, quotes that recur within a document
- **Neural network** handles everything else: semantics, long-range dependencies, novel compositions

By offloading predictable and repetitive patterns to specialized mechanisms, the transformer's learned parameters are freed to focus on the genuinely hard predictions.

## Configuration

| Parameter | Value |
|-----------|-------|
| model_dim | 640 |
| num_layers | 11 |
| num_heads | 10 |
| num_kv_heads | 5 |
| mlp_mult | 2 |
| vocab_size | 1024 |
| train_seq_len | 1024 |
| train_batch_tokens | 524,288 |
| tie_embeddings | yes |
| copy_net | yes |
| dict_residual | yes |

## Reproduction

```bash
RUN_ID=copy_brotli_seed1337 \
COPY_NET=1 \
DICT_RESIDUAL=1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For local testing on 1xGPU:
```bash
NUM_LAYERS=5 TRAIN_SEQ_LEN=512 TRAIN_BATCH_TOKENS=32768 \
COPY_NET=1 DICT_RESIDUAL=1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Novel Contributions

1. **Copy Network for parameter-constrained LM**: Applying pointer networks / copy mechanisms to the parameter golf setting. No prior submission has used this approach.

2. **Dictionary Residual for LM**: Using a frozen n-gram dictionary as a base predictor with neural correction. Directly inspired by dictionary-based compression (Brotli/zstd). No prior submission has used this approach.

3. **Compression-theory-motivated architecture**: Both mechanisms are grounded in classical information theory -- LZ77 backreferencing and dictionary compression -- rather than being ad-hoc neural architecture modifications.

## Included Files

- `train_gpt.py` -- complete training and evaluation script
- `README.md` -- this file
- `submission.json` -- leaderboard metadata (added after verification)
- `train_seed*.log` -- training logs (added after verification)
