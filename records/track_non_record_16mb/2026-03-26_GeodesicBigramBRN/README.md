# Geodesic Topological Tokenizer + BigramHash + BRN

**Non-Record Submission (Research Contribution)**
**Author:** skar07 ([@skar07](https://github.com/skar07))
**Track:** non_record_16mb
**Final Results (Preliminary):** 1.5406 BPB | **Validation Loss:** 3.4562
**Comparison:** +0.4596 vs Rank 1 (1.0810)

---

## 🔬 Research Overview

This project explores a multi-pronged approach to attacking the Bits-Per-Byte (BPB) bottleneck in parameter-constrained transformers. Instead of relying on standard BPE, we introduce a novel **Geodesic Topological Tokenizer** that leverages the manifold structure of language.

> [!NOTE]
> This is a research-oriented submission. I'll be updating the components and results as the investigation progresses. Preliminary tests on Colab T4 showed a BPB of ~1.53, with recent 8xH100 runs converging to 1.5406.

---

## 🛠️ The Architecture

### 1. Geodesic Topological Tokenizer
The core innovation is replacing the standard SentencePiece BPE with a tokenizer built from **persistent homology** on n-gram co-occurrence manifolds.

*   **Manifold Projection:** Character n-grams are projected onto a geodesic manifold using **Isomap**.
*   **Topological Clustering:** Finds topologically stable clusters on the manifold.
*   **Compression Advantage:** Achieves **3.236 chars/token**, a significant leap over SP1024 BPE's **2.24 chars/token** (~44% advantage).
*   **Status:** Currently trained on **wikitext** and used to tokenize **fineweb**. There is likely further headroom if the tokenizer is trained directly on FineWeb shards.

### 2. BigramHash
Consecutive token pairs are hashed to a learned embedding table and gated into the input representation before attention. This provides the model with "free" bigram statistics as a sequence-level prior, reducing the number of parameters needed to learn local dependencies.

### 3. BRN (Belief Recurrent Network)
Maintains a persistent **belief state** across the sequence. This state gates how much new information updates the representation and is broadcast back to all token positions, acting as a dynamic sequence-level prior.

---

## 📊 Training & Performance

| Metric | Value |
| :--- | :--- |
| **Parameters** | 25,710,664 |
| **Steps** | 50,000 |
| **Training Time** | ~165 minutes |
| **Hardware** | 8xH100 (TBD optimization) |
| **Final BPB** | 1.5406 |
| **Val Loss** | 3.4562 |
| **Rank 1 Delta** | +0.4596 (vs 1.0810) |

### Tokenizer Efficiency Comparison
| Method | Chars per Token | Advantage |
| :--- | :--- | :--- |
| **Geodesic (Ours)** | **3.236** | **Base** |
| SP1024 BPE | 2.24 | -44% |

---

## 🚀 Future Work
- [ ] Train Geodesic Tokenizer on FineWeb shards for better alignment.
- [ ] Optimize BRN state update logic for faster wall-clock throughput.
- [ ] Explore higher-order topological features (cycles, voids) for token separation.

---

## 📜 Acknowledgments
Special thanks to the OpenAI `parameter-golf` community for the inspiration and references. This work builds upon topological research in transformers.
