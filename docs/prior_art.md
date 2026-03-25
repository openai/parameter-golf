# Prior Art Analysis: BESE Tokenizer

## Summary

After thorough research, the specific combination of techniques in BESE appears to be novel.
The individual components have precedents, but nobody has combined them in this way for LLM
tokenization. Here is what exists and what doesn't.

---

## What HAS been done (related work)

### 1. Hierarchical Autoregressive Transformers (ICLR 2025)
- **Paper:** "Combining Byte- and Word-Level Processing for Robust, Adaptable Language Models"
- **What it does:** Uses character-level encoder to build word embeddings, then a word-level backbone
- **Similarity to BESE:** Two-layer hierarchy (characters -> words), eliminates fixed vocabulary
- **Key difference:** Their base alphabet is raw UTF-8 bytes (256 tokens). BESE uses a DESIGNED
  38-token alphabet with frequency weighting and context-aware grouping. They don't restructure the
  base alphabet itself.

### 2. Dynamic Grouping with Hierarchical BPE (EMNLP 2025 Findings)
- **Paper:** "From Characters to Tokens: Dynamic Grouping with Hierarchical BPE"
- **What it does:** Applies BPE hierarchically with dynamic grouping of characters
- **Similarity to BESE:** Hierarchical BPE, grouping characters before merging
- **Key difference:** Their grouping is learned/dynamic, not designed based on co-occurrence analysis.
  They use standard byte or character base, not a frequency-weighted structured alphabet.

### 3. T-FREE: Tokenizer-Free via Sparse Representations (2024)
- **Paper:** "T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations"
- **What it does:** Replaces tokenization with character trigram hashing. Achieves 85%+ embedding
  compression.
- **Similarity to BESE:** Same goal (massive embedding parameter savings). Also moves away from
  standard BPE vocabulary.
- **Key difference:** Completely different mechanism (sparse hashing vs. structured variable-length
  encoding). No frequency-weighted codes, no context-aware grouping.

### 4. MegaByte (Meta, 2023)
- **Paper:** "MegaByte: Predicting Million-Byte Sequences with Multiscale Transformers"
- **What it does:** Hierarchical transformer that processes raw bytes with a local+global architecture.
- **Similarity to BESE:** Addresses the byte-level sequence length problem with hierarchy.
- **Key difference:** Architectural solution (modified transformer), not a tokenizer redesign.
  Still uses raw bytes as base vocabulary.

### 5. Efficient Vocabulary Reduction (COLING 2025)
- **Paper:** "Efficient Vocabulary Reduction for Small Language Models"
- **What it does:** Reduces existing large vocabularies (128K -> 8K) to save embedding parameters.
- **Similarity to BESE:** Same motivation (embedding table is too large for small models).
- **Key difference:** Top-down approach (pruning existing vocab) vs. BESE's bottom-up approach
  (building from structured alphabet). They still use standard BPE as the base.

### 6. Case-Insensitive Tokenizers (Standard Practice)
- BERT uncased models use lowercase-only tokenization. This is well-established.
- **Similarity to BESE:** Dropping case for parameter efficiency.
- **Key difference:** BERT uncased is a preprocessing step on standard BPE/WordPiece. BESE
  integrates case elimination into the alphabet design itself.

### 7. Huffman Coding for Neural Networks (Various)
- Using variable-length codes based on frequency is ancient (Huffman, 1952).
- Applied to neural network weight compression extensively.
- **Similarity to BESE:** Frequency-weighted encoding.
- **Key difference:** Huffman coding has been applied to weights and data compression, but not
  to designing a base alphabet for BPE tokenization of LLM training data.

### 8. Binary BPE (2025)
- **Paper:** "Binary BPE: A Family of Cross-Platform Tokenizers for Binary Analysis"
- **What it does:** Applies BPE to raw binary (non-text) data using the 256-byte alphabet.
- **Similarity to BESE:** BPE on a non-standard domain.
- **Key difference:** Still uses the standard 256-byte base. The novelty is the domain (binary
  executables), not the base alphabet design.

---

## What has NOT been done (BESE's novel contributions)

### 1. Designed base alphabet for BPE: NOVEL
Nobody has replaced the standard 256-byte base alphabet with a PURPOSE-DESIGNED smaller alphabet
before running BPE. Every BPE implementation starts from either:
- 256 raw bytes (GPT-2, GPT-4, Llama)
- Unicode characters (original BPE)
- Domain-specific raw tokens (MIDI, binary)

BESE starts from 38 tokens designed around English letter frequency. This is genuinely new.

### 2. Frequency-weighted variable-length character codes as BPE base: NOVEL
The idea that common letters (e, t, a, o, i, n, s, r) should be single tokens while rare letters
(z, q, x, j) should be multi-token codes, and then running BPE on top of this structured encoding,
has no precedent in the literature I found.

### 3. Context-aware character grouping (QWERTY-inspired): NOVEL
Using bigram co-occurrence analysis to determine which letters share a group token (minimizing
within-group context similarity so the model can disambiguate from context) has not been applied
to tokenizer design. The QWERTY connection (separating co-occurring letters) is a unique framing.

### 4. The specific combination: NOVEL
Even if individual pieces have loose precedents, the combination of:
  - Frequency-weighted variable-length character encoding
  - Context-aware grouping based on bigram analysis
  - Case-insensitive encoding for parameter efficiency
  - BPE merges on top of the structured alphabet
...has not been published or proposed anywhere I can find.

---

## Closest overall work

**T-FREE (2024)** is the closest in SPIRIT: both approaches aim to massively reduce embedding
parameters by replacing standard tokenization with something more parameter-efficient. But the
MECHANISMS are completely different (sparse trigram hashing vs. structured variable-length encoding
with BPE).

**Hierarchical Autoregressive Transformers (ICLR 2025)** is the closest in STRUCTURE: both use
a two-layer hierarchy. But they don't redesign the base alphabet, they just add a word-level
layer on top of standard bytes.

---

## Conclusion

The BESE approach is genuinely novel in its specific combination of techniques. The individual
inspirations (Huffman coding, bigram analysis, case folding, BPE) are all well-established, but
their synthesis into a designed base alphabet for parameter-constrained LLM tokenization appears
to be new. This should be clearly stated in the submission write-up, with appropriate citations
to the related work above.

The honest framing: "We apply well-known information-theoretic principles (variable-length coding,
mutual information minimization) in a new context (designing the base alphabet for BPE tokenization
under extreme parameter constraints) and combine them with BPE merges to achieve embedding savings
while maintaining competitive sequence lengths."
