# Competitive Stack + Phonetic Tokenization Exploration

**Non-record submission | val_bpb: 1.2055 | 4xH100 SXM**

## Approach

Two contributions: a competitive training stack combining top leaderboard techniques, and an exploration of phonetic (IPA) tokenization as a novel alternative to BPE.

### Training Stack

| Technique | Description |
|-----------|-------------|
| Int6 STE QAT | Fake-quantize weights to 6-bit during training, straight-through gradients |
| BigramHash | 4096-bucket hash embedding for bigram context (~590K params) |
| SmearGate | Learned gate blending current/previous token embeddings |
| OrthoInit | Orthogonal weight init with muP-style 1/sqrt(2L) projection scaling |
| MLP 3x | Wider feedforward (1536 hidden dim) |
| Sliding Window Eval | Overlapping windows at stride=64 |

Tuned hyperparameters: `lr=0.02`, `momentum=0.99`, `warmdown=3000`, `grad_clip=0.3`, `batch=786K`, `seq_len=2048`.

### Phonetic Tokenization Exploration

We tested whether converting text to IPA (International Phonetic Alphabet) before tokenization could help compression. The idea: English spelling is irregular ("knight" vs "night" look different but sound the same), so a phonetic representation might be more learnable.

We built a G2P converter with 4795 CMUdict exceptions (84.6% FineWeb word coverage) and trained SentencePiece BPE on the phonetic output. In a controlled 1xH100 comparison:

| Tokenizer | val_bpb |
|-----------|---------|
| Standard BPE-1024 | 1.335 |
| IPA-BPE-1024 | 1.329 |

IPA-BPE showed a small gain (0.006 bpb), suggesting phonetic structure does encode some useful signal. However, the 7% token expansion means fewer training steps per time budget, which likely offsets this advantage at scale.

**Interesting negative result**: the competitive stack techniques (BigramHash, SmearGate, etc.) appear to capture the same local patterns that phonetic encoding provides, making the IPA preprocessing redundant when combined with modern techniques.

## Results

| Metric | Value |
|--------|-------|
| val_bpb (standard) | 1.2262 |
| **val_bpb (sliding)** | **1.2055** |
| Steps | 3509 |
| Hardware | 4xH100 SXM |
| Model size | 19.6MB (int8+zlib) |

## Known Issues

1. **Over 16MB**: Needs int6 export quantization (currently int8).
2. **4xH100**: Not the required 8xH100 for record submissions.

## Command

```bash
VOCAB_SIZE=1024 TRAIN_SEQ_LEN=2048 MLP_MULT=3 QAT=1 ORTHO_INIT=1 \
USE_BIGRAM_HASH=1 USE_SMEAR_GATE=1 MATRIX_LR=0.02 SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
TRAIN_BATCH_TOKENS=786432 EVAL_STRIDE=64 EVAL_BATCH_SEQS=1024 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```
