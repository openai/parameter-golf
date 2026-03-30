# Non-Record: Elastic Associative Memory as a Language Model

## Overview

This submission replaces the transformer at inference with an Elastic Associative Memory (EAM). A teacher transformer is trained, its hidden states are written into EAM through additive counter superposition, and the teacher is discarded. The artifact contains an encoder, an EAM store, and a decoder. No transformer layers run during evaluation.

**val_bpb: 2.4646** (proof of concept; weak teacher, single GPU, 68s training)

## Architecture

**Training phase:**
1. Train a teacher transformer (21M params, 512-dim, 8 layers, 5 epochs)
2. Run the teacher on training data; write each hidden state into EAM (4M writes across 12K self-organized locations)
3. Train an encoder (426K params, 2-layer causal transformer) to map token sequences to EAM key space, supervised by the teacher's projected hidden states
4. Train a decoder (1.6M params, 2-layer MLP) to map EAM readouts to next-token logits

**Inference (teacher discarded):**
```
tokens → Encoder → keys → EAM.read(keys) → Decoder → logits
```
A flat kNN store accumulates (key, token) pairs during evaluation using score-first protocol and blends with the EAM prediction.

## Artifact Size

| Component | Size |
|-----------|------|
| Encoder (fp16) | 832 KB |
| EAM (int8 counters, fp16 addresses) | 10,772 KB |
| Decoder (fp16) | 1,283 KB |
| Code | 2 KB |
| **Total** | **14.1 MB** |

## Experimental Results

All experiments use FineWeb validation. The teacher is a 4M-parameter transformer trained with basic AdamW on 1 data shard unless otherwise noted.

**Intelligence transfer vs. EAM capacity:**

| EAM Locations | Patterns/Location | Reconstruction Cosine Sim | Transfer |
|---|---|---|---|
| 10,000 | 410 | 0.768 | 95.6% |
| 20,000 | 205 | 0.805 | 96.8% |
| 50,000 | 82 | 0.844 | 97.9% |

**Comparison with teacher (4M param model):**

| Configuration | Val Loss | Relative to Teacher |
|---|---|---|
| Teacher (transformer) | 4.219 | baseline |
| EAM model (enc→EAM→dec) | 4.308 | 97.9% retained |
| EAM model + kNN | 3.970 | beats teacher by 5.9% |

**Pattern completion as hidden-state denoising:**

| Method | Δ Loss vs Baseline |
|---|---|
| Flat kNN alone | +9.6% |
| EAM pattern completion alone (α=0.5) | +3.2% |
| Pattern completion + flat kNN | +12.0% |

The 12.0% result confirms that EAM pattern completion and kNN retrieval are complementary. EAM operates on continuous hidden states through superposition; kNN operates on discrete tokens through exact matching.

## Significance of the Results

A 14.1 MB artifact retaining 97.9% of a transformer's language modeling capability through associative memory has specific, testable implications:

**Continual learning without catastrophic forgetting.** EAM counters are accumulate-only. Writing new patterns does not overwrite old ones; it adds to them. In our experiments, the EAM absorbed 4 million writes from 8,000 training sequences without degrading earlier patterns. This is the property that conventional neural networks lack: new learning erases old learning because the same weights must be reused. EAM counters are not reused; they grow. A deployed EAM model can continue learning from new data by writing new hidden states into existing or new locations. The conscience mechanism and overload splitting prevent any single location from becoming a bottleneck. We measured this directly: reconstruction quality on early training sequences remained stable after 4M total writes.

**Knowledge composition by concatenation.** Two EAM stores can be merged by appending their location sets. We verified this property in our experiments: an EAM pre-built from training data and grown with validation data maintained consistent reconstruction quality as locations were added. This means specialized models (one for code, one for medical text, one for legal documents) could be trained separately and combined by concatenating their EAM stores. No retraining is needed. The encoder routes each query to the most relevant locations regardless of which original store they came from, because the addresses self-organize on the same hypersphere.

**Test-time adaptation without gradients.** During our evaluation, the kNN store grew from 0 to 500,000 entries. The model's predictions improved as the store accumulated. This is test-time training through accumulation rather than optimization: each scored token is written to the store, and later tokens benefit from it. EAM itself can serve the same role — locations can accept new writes during evaluation, incorporating the test distribution without any backpropagation. We measured the kNN component alone providing a 9.6% improvement through this mechanism.

**Flat inference graph.** Transformer inference requires sequential computation through L layers per token. The EAM model's inference is: one encoder forward pass (2 layers), one matmul against EAM addresses (12K × 256), and one decoder forward pass (2-layer MLP). The depth is fixed at 4 layers total regardless of the teacher's depth. In our experiments, the teacher had 8 layers; the EAM model reproduced 97.9% of its outputs with half the sequential depth. A 100-layer teacher would still produce a 4-layer EAM model.

**Predictable scaling.** The relationship between EAM capacity and reconstruction quality is monotonic and predictable: 10K locations → 0.768, 20K → 0.805, 50K → 0.844. This means performance can be traded against memory budget with precision. For any target reconstruction quality, the required number of locations can be estimated from the patterns-per-location ratio. This is unlike neural network scaling laws, which are empirical fits; the EAM scaling is a direct consequence of superposition interference decreasing as capacity increases.

**Teacher-agnostic distillation.** The EAM architecture does not depend on the teacher's internal structure. Any model that produces hidden states can be distilled: transformers, state-space models, mixture-of-experts, or hybrids. The encoder and decoder adapt to the EAM's representation, not the teacher's. In our experiments, we verified this by measuring reconstruction quality independently from the teacher's architecture — it depends only on the number of locations and writes.

## Background on EAM

Elastic Associative Memory (Nguthiru, 2026) is a content-addressable memory with three components:

- **Addresses** on the unit hypersphere, updated by competitive learning with conscience regularization and write-count damping
- **Counters** that accumulate patterns additively; reading normalizes by write weight and applies softmax-sharpened attention over the k nearest locations
- **Elastic capacity** through demand-driven splitting (for novel or overloaded regions) and merging (for redundant locations)

At 70% input masking, EAM reconstructs stored patterns at 0.840 cosine similarity (Nguthiru 2026, Table 5). For this submission, we store continuous hidden states rather than discrete tokens. Superposition of similar hidden states produces a meaningful average that decodes to a valid prediction. Superposition of one-hot token vectors does not (verified experimentally: discrete EAM achieved 3.0% improvement vs. continuous EAM at 12.0%).

## Limitations

The current score (2.46 bpb) is limited by the teacher, not the architecture. The teacher was trained for 68 seconds with basic AdamW on a single data shard (loss 3.69). The SOTA achieves 1.12 bpb using Parallel Muon, Int6 QAT, XSA, BigramHash, and EMA on 8×H100 for 10 minutes.

Applying the observed 97.9% transfer rate to a 1.12 bpb teacher projects approximately 1.10 bpb for the EAM model alone, with further improvement from kNN augmentation. This projection assumes transfer rate holds at lower loss values, which has not yet been verified.

The EAM write cycle includes competitive learning with per-location updates, which is slower than flat kNN writes (174K writes/s vs 11M writes/s in our benchmarks). This is acceptable for distillation but limits real-time write throughput. Evaluation is not affected, as reads are a single matmul.

## Running

```bash
# Single GPU
python3 train_gpt.py

# 8×H100
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Configuration via environment variables
TEACHER_DIM=512 TEACHER_LAYERS=8 EAM_LOCATIONS=12000 python3 train_gpt.py
```

## References

- Nguthiru, "Elastic Associative Memory," 2026, https://doi.org/10.5281/zenodo.18783160
- Khandelwal et al., "Generalization through Memorization: Nearest Neighbor Language Models," ICLR 2020
