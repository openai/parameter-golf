# GPT-2 Inspired Model Optimization for the 10-Minute, 16MB Track

**Author:** Quân group HCMUS

## 1. Introduction
This submission aims to maximize language modeling performance under the strict constraints of the `track_10min_16mb` challenge. Achieving top-tier Byte-Per-Token (BPB) scores within a maximum of 10 minutes of wall-clock training time and a strict 16.0 MB storage limit for the final serialized artifact requires aggressive optimization across three fronts: architecture layout, training dynamics, and post-training quantization. This approach explores how parallelizing residual streams and effectively routing information deep into the network can push the boundaries of cross-entropy loss under these extreme constraints.

## 2. Architectural Innovations

### 2.1. U-Net Style Skip Connections
To alleviate vanishing gradients and permit the free flow of low-level linguistic features into deep layers, the architecture implements U-Net style skip connections. 
- The network is logically split into an encoder half and a decoder half.
- State tensors from the encoder phase (`skips`) are cached and directly injected back into the decoder phase. 
- A learnable scalar gating mechanism (`skip_gates`) employs a sigmoid activation to dynamically interpolate between the deep representation and the injected shallow representation. This ensures that the model only utilizes the shallow information when beneficial.

### 2.2. Parallel Residuals (Late-Stage Parallelism)
Standard Transformer architectures process the Self-Attention and MLP blocks sequentially. This limits GPU utilization and increases deep-graph latency. 
- In this model, starting at `PARALLEL_START_LAYER = 7`, the processing splits into two independent computing lanes: `lane0` (Attention) and `lane1` (MLP).
- Both lanes receive identical initial states (incorporating the U-Net skip delta) and compute their respective functions in parallel.
- A learnable parameter (`lane_merge`) smoothly merges `lane0` and `lane1` back into a single sequence before the final normalization layer, significantly reducing depth latency while increasing effective width.

### 2.3. Depth Recurrence
To exponentially increase the perceptual depth of the network without inflating the parameter count, dynamic iteration of block subsets is employed.
- `RECUR_LAYERS = [3, 4, 5]`: These specific middle layers are physically traversed multiple times in a single forward pass.
- Recurrence is turned off in the beginning to maintain stable initial convergence and is toggled on seamlessly at `RECUR_START_STEP = 3000`.

### 2.4. Value Embedding Enhancements (VE)
The `ve_enabled` toggle augments the Key-Value (KV) cache with extra representations injected directly into the Multi-Head Attention layer. This permits isolated layers (`ve_layers = 9, 10`) to retrieve rich spatial metadata without dedicating MLP capacity to routing these signals.

## 3. Training & Optimization Strategy

The training routine employs a heterogeneous optimization strategy, carefully dividing variables by their geometric properties:
1. **Muon Optimizer for Matrix Weights**:
   - Applies to strictly 2-Dimensional weight tensors (e.g., projections, linear layers).
   - Uses `Newton-Schulz 5` (NS5) steps to iteratively orthogonalize the gradients, forcing updates that decorate the matrix manifold efficiently.
   - Operates with an aggressive momentum of `0.99` (warmed up from `0.92` over 1500 steps) and a weight decay of `0.095`.
2. **AdamW for Vectors & Scalars**:
   - Used for embeddings, biases, layer norms, and scalar variables (e.g., skip weights/gates).
   - Utilizes standard heuristics (β1 = 0.9, β2 = 0.95), EPS of 1e-8, and lower learning rates (`SCALAR_LR = 0.02`).

**Data Loading & Context**:
- Uses coprime-stride multi-shard token loaders.
- Global Sequence Length of `2048` tokens for both training and validation.
- Early stopping is rigorously applied directly through `MAX_WALLCLOCK_SECONDS` ensuring training never violates the 10-minute rule.

## 4. Post-Training Quantization (16MB Target)

Achieving a sub-16MB footprint from a ~74M parameter footprint necessitates brutal compression.
1. **GPTQ with ActOrder**: We utilize second-order information (Hessian matrices inverted via Cholesky decomposition) tracked over 64 calibration batches. Weights are quantized down to INT6 precision per-row with clipping optimization spanning various percentiles (from 99.9% to 100%).
2. **Selective ±1 Pruning**: After primary quantization, the overall size is evaluated. If it exceeds 16,000,000 bytes (inclusive of source code), the algorithm targets the ±1 quantized states with the lowest scaled error impact and zero-prunes them, mathematically guaranteeing the artifact fits.
3. **Brotli Transposition Compression**: A custom `_byte_shuffle` stride alignment followed by Level 11 Brotli compression compresses the bit-packed representation exceptionally well.

## 5. Evaluation & Sliding Window Inference
Measurement is handled via a strided contextual window evaluation (`EVAL_STRIDE = 64`). 
- Instead of block-scoring, which artificially penalizes tokens at the edge of the context window, `eval_val_sliding` scores each token using the maximal historical context available up to `seq_len(2048)`.
- It executes a `torch.compile` (`dynamic=False, fullgraph=True`) graph to leverage TensorRT optimizations dynamically.

---

## 6. Official Metrics

Results aggregated over 3 unique pseudo-random seeds (`1337`, `42`, `1024`).

| Metric | Aggregate Average |
| :--- | :--- |
| **Max Training Steps** | ~5080 |
| **Val Loss (Cross Entropy)** | 2.5084 |
| **Val BPB (Byte-Per-Token)** | 1.0901 |
| **Artifact Size (Bytes)** | 15,976,317 |

*Note on BPB: The tokenization operates using a strictly configured Byte-Pair Encodings (BPE) vocab of 4096 spanning the FineWeb dataset segments. 1.09 BPB demonstrates that the combination of Deep Recurrence and Parallelized Skips is extracting nearly maximum informational entropy for this subset.*

## 7. Hyperparameter Configuration

The model's superior BPB performance is highly sensitive to its hyperparameter tuning. The following parameters dictate the capacity and training dynamics:

### 7.1. Architectural Dimensions
- **Layers (`NUM_LAYERS`)**: 11
- **Hidden Dimension (`MODEL_DIM`)**: 512
- **Attention Heads (`NUM_HEADS`)**: 8
- **Key-Value Heads (`NUM_KV_HEADS`)**: 4 (employing Grouped-Query Attention constraints to preserve bandwidth)
- **Vocabulary Size (`VOCAB_SIZE`)**: 4096 (Custom Byte-Pair Encoding mapping)
- **Global Sequence Length (`SEQ_LEN`)**: 2048

### 7.2. Routing & Structural Hyperparameters
- **`PARALLEL_START_LAYER`**: 7. Layers 0-6 remain fully sequential, allowing the model to build foundational linguistic representations before splitting into the `lane0` (Attention) and `lane1` (MLP) dual-processing streams.
- **`RECUR_LAYERS`**: `[3, 4, 5]`. These middle layers are dynamically traversed multiple times.
- **`RECUR_START_STEP`**: 3000. Recurrence is disabled initially and toggled on dynamically after substantial warm-up to prevent representation collapse early in training.
- **`VE_LAYERS` (Value Embedding)**: `[9, 10]`. High-level layers where KV caching is supplemented by extra spatial representations.

### 7.3. Optimization & Regularization
- **Muon Settings**: Internal weight matrices utilize continuous orthogonalization steps with a `momentum` of `0.99` and an aggressive geometric `weight_decay` of `0.095`.
- **Learning Rates**:
  - `MATRIX_LR`: 0.022 (For Muon parameters)
  - `EMBED_LR`: 0.6 (For heavily scaled embeddings)
  - `SCALAR_LR`: 0.02 (For AdamW parameters like layout bounds and skip gates)
  - `MIN_LR`: 0.0
- **Warmdown Fraction**: 0.667 (66.7% of the total optimization timeline is dedicated to a structured cooldown phase).

### 7.4. Evaluation Setup
- **`EVAL_STRIDE`**: 64. Applied during `eval_val_sliding` to maximize evaluation context utilization for edge tokens, completely evading chunk-based penalization.
