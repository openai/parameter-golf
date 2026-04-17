# Non-Record Submission: V5 SP1024 + Seq4096 (1xH100)

## Overview

This is a **non-record submission** for the OpenAI Parameter Golf challenge.

The run was executed on the official FineWeb `sp1024` dataset using an extended sequence length (`4096`) and a single H100 GPU.  
It does **not qualify for record track**, as it was not trained under the official constraint of **10 minutes on 8×H100**.

---

## Key Results

- **Final val_bpb:** 1.21430168
- **Final val_loss:** 2.05029752
- **Total submission size:** 15,841,388 bytes
- **Training steps:** 6000
- **Wallclock time:** ~3218 seconds
- **GPU:** 1×H100

---

## Training Configuration

- Dataset: FineWeb (sp1024 tokenizer)
- Train sequence length: 4096
- Train batch tokens: 524,288
- Gradient accumulation: 8
- Warmup steps: 30
- Seed: 1337

Training follows the official `train_gpt.py` pipeline with modifications to support longer context.

---

## Model Architecture

- Parameters: 17,059,912
- Layers: 9
- Model dimension: 512
- Attention heads: 8
- KV heads: 4 (GQA)
- MLP multiplier: 2
- Embeddings: tied

Architecture is based on a compact transformer optimized for parameter efficiency and compression.

---

## Optimization Details

- Optimizer:
  - Muon (matrix parameters)
  - Adam (embeddings and scalar parameters)
- Learning rate scheduling:
  - Warmup + wallclock-based decay
- Mixed precision: bfloat16
- Flash attention enabled

---

## Compression & Serialization

Model is exported using:

- **int8 quantization (per-row for matrices)**
- **zlib compression**

Final artifact:
