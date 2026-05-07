# Parameter Golf Submission: FastClusterGating 128Cluster

This repository contains the official submission for the **OpenAI Parameter Golf Challenge**.

The **FastClusterGating 128Cluster** model introduces an alternative architectural approach focused on *intelligence density* — maximizing learning efficiency within strict compute and size constraints.

---

## 🚀 Architecture Overview

- **Model Type**: Transformer
- **Layers**: 8
- **Hidden Dimension**: 640
- **Attention Heads**: 10
- **Sequence Length**: 1024
- **Clusters**: 128

### Cluster Specialist Core

Instead of relying purely on standard transformer layers, this model integrates a **cluster-based gating mechanism**:

- **FastClusterGating** dynamically routes token representations through learned soft clusters.
- Clusters specialize on token-level patterns and contextual nuances.
- Enables fine-grained biasing of logits without increasing core model size.

This allows the model to approximate higher-capacity behavior while remaining within a strict parameter budget.

---

## ⚙️ Optimization Strategy

- **Muon Optimizer**
  - Applied to matrix parameters
  - Enforces structured, orthogonal updates

- **Adam Optimizer**
  - Used for embeddings, scalars, and cluster parameters

- **Training Precision**
  - `bfloat16` for stability and performance on H100 hardware

- **SWA (Stochastic Weight Averaging)**
  - Applied during late-stage training for improved generalization

---

## 📊 Final Performance (8× H100 SXM)

| Metric | Value |
|--------|------|
| **Training BPB (Final)** | **1.2251** |
| **Quantized Roundtrip BPB** | **1.2331** |
| **Compressed Model Size** | **13.00 MB** |
| **Training Time** | **~589 seconds (within 600s cap)** |
| **Total Runtime** | **~616 seconds** |

All results were obtained under official competition constraints.

Validation and final metrics are documented in `training_log.txt`.

---

## 📦 Submission Contents

- `train_gpt.py`  
  Complete standalone script for:
  - Model definition  
  - Training loop  
  - Evaluation  
  - Quantization  

- `final_model.int8.ptz`  
  Compressed model artifact using mixed-bit quantization:
  - uint5 / uint6 / int8
  - zlib compressed

- `training_log.txt`  
  Full execution trace of the final run

---

## 🧠 Design Philosophy

The **FastClusterGating 128Cluster** model is built around a key principle:

> **Efficient specialization beats uniform scaling.**

By allocating representational capacity dynamically through clustering, the model:

- Reduces redundancy in learned representations
- Improves token-level precision
- Maintains competitive BPB within strict constraints

---

## 📏 Rule Compliance

This submission strictly adheres to all Parameter Golf rules:

- ✔ No n-gram or lookup-based shortcuts
- ✔ No test-time training (TTT)
- ✔ Pure neural architecture
- ✔ ≤ 600s training time
- ✔ ≤ 16MB total submission size

---

## 🛠️ Running the Model

Set required environment variables:

```bash
DATA_PATH="/path/to/fineweb10B_sp1024"
TOKENIZER_PATH="/path/to/tokenizer.model"
torchrun --standalone --nproc_per_node=8 train_gpt.py
