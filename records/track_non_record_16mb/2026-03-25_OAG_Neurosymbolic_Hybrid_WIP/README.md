# Non-record WIP: OAG Neurosymbolic Hybrid

## Approach
Ontology-Augmented Generation (OAG) applied to text compression. Five complementary predictors blended at eval-time with zero artifact cost:

### Architecture

```
                    +-----------------+
                    | Neural Model    |  (SOTA: LeakyReLU^2 + TTT + Parallel Muon)
                    | (scored via CE) |
                    +--------+--------+
                             |
                       [raw logits]
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
     softmax(logits)   entropy(logits)   context tokens
            |                |                |
            |     +----------+----------+     |
            |     |                     |     |
            v     v                     v     v
    +-------+-----+--+  +---------+  +-+-----+-------+  +--------+
    | FST Predictor  |  | N-gram  |  | Match Model   |  | Neural |
    | (structural    |  | Cache   |  | (LZ77-style   |  | probs  |
    |  patterns)     |  | (ord-8  |  |  substring)   |  |        |
    |                |  |  MKN)   |  |               |  |        |
    +-------+--------+  +---+-----+  +-------+-------+  +---+----+
            |                |                |              |
            v                v                v              v
         +--+----------------+----------------+--------------+--+
         |              GLN Context Mixer                       |
         |  (Gated Linear Network, O(log T) regret)            |
         |  [context = neural entropy -> 8 discrete bins]       |
         |  [exponentiated gradient weight updates]             |
         +---------------------------+--------------------------+
                                     |
                               [mixed probs]
                                     |
                        score token (NLL from mixture)
                                     |
                        observe token -> cache, match
                        (backward-looking: score FIRST)
```

### Components

1. **Neural Transformer** (SOTA stack) -- semantic understanding, base predictor
2. **FST Grammar Predictor** (`fst_predictor.py`) -- deterministic structural prediction
   - 67 HTML tags, 73 boilerplate phrases, 17 common phrases
   - URL/JSON/date pattern matching
   - ~1.3% high-confidence coverage on FineWeb
3. **N-gram Cache** (`ngram_cache.py`) -- adaptive local pattern memorization
   - Order-8 (upgraded from order-5)
   - Modified Kneser-Ney smoothing (Chen & Goodman, 1999)
   - Continuation count-based lower-order distributions
   - 95.8% coverage, backward-looking only
4. **Match Model** (`match_model.py`) -- LZ77-style substring matching
   - Finds longest matching context in already-scored tokens
   - Predicts continuation from match history
   - Suffix-indexed for efficient O(max_order) lookup
   - Zero artifact cost (no learned parameters)
5. **GLN Context Mixer** (`gln_mixer.py`) -- provably optimal online mixing
   - Gated Linear Network (Veness et al., 2021)
   - Per-context-bin exponentiated gradient updates
   - O(log T) regret vs best fixed expert combination
   - Context = discretized neural entropy (8 bins)
   - Fallback to simple alpha-blending during warmup (first 500 tokens)

### Integration

- `hybrid_eval.py`: Unified eval function `eval_val_sliding_hybrid()` that replaces `eval_val_sliding()`
- Activated by `HYBRID_EVAL=1` environment variable
- Training is completely unchanged -- all innovations are eval-time only
- All predictors are strictly backward-looking (observe AFTER scoring)

## Academic References

- **Kneser-Ney smoothing**: Chen & Goodman (1999), "An Empirical Study of Smoothing Techniques for Language Modeling", Computer Speech and Language 13(4)
- **Gated Linear Networks**: Veness et al. (2021), "Gated Linear Networks", AAAI 2021
- **LZ77 matching**: Ziv & Lempel (1977), "A Universal Algorithm for Sequential Data Compression", IEEE Trans. Information Theory
- **Context Tree Weighting**: Willems et al. (1995), "The Context-Tree Weighting Method: Basic Properties", IEEE Trans. Information Theory
- **Entropy-adaptive mixing**: Similar to PPM/PAQ prediction by partial matching

## Preliminary Results (1xH100, partial training)

| Config | BPB | Delta |
|--------|-----|-------|
| Neural only | 4.5070 | baseline |
| + Cache gentle (order-5) | 4.4905 | -0.0165 |
| + FST gentle | 4.5062 | -0.0008 |
| + Both gentle | **4.4900** | **-0.0170** |

Upgrades in this version (not yet validated on 8xH100):
- N-gram cache: order-5 -> order-8, simple interpolation -> Modified Kneser-Ney
- New: LZ77 match model for long-range substring prediction
- New: GLN context mixer replacing simple alpha-blending
- Expected additional improvement: 0.02-0.05 BPB from better mixing + match model

## Competition Legality

All eval-time predictors are confirmed legal per competition rules:
- N-gram cache: backward-looking only (confirmed legal by organizers)
- Match model: backward-looking only (same principle as n-gram cache)
- FST: grammar knowledge only, no learned parameters
- GLN mixer: online learning from scored tokens only
- Entropy-adaptive alpha: uses model uncertainty, never ground truth

## Usage

```bash
# Standard eval (no hybrid)
torchrun --nproc_per_node=8 train_gpt.py

# With hybrid eval
HYBRID_EVAL=1 torchrun --nproc_per_node=8 train_gpt.py
```

## Files

| File | Description | Artifact Cost |
|------|-------------|---------------|
| `train_gpt.py` | SOTA training script + hybrid eval integration | 0 (eval-only code) |
| `fst_predictor.py` | Finite State Transducer for web text patterns | 0 |
| `ngram_cache.py` | Order-8 n-gram cache with Modified Kneser-Ney | 0 |
| `match_model.py` | LZ77-style longest substring match predictor | 0 |
| `gln_mixer.py` | Gated Linear Network context mixer | 0 |
| `hybrid_eval.py` | Unified eval function integrating all predictors | 0 |

## Status
WIP. Requesting 8xH100 compute for full evaluation with all predictors active.
