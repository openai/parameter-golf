# Parameter Golf Mastery Curriculum

A semester-long (16-week) curriculum covering every discipline required to dominate the OpenAI Parameter Golf challenge. Synthesized from the pedagogical traditions of CMU (systems + optimization), Stanford (deep learning theory + scaling), MIT (information theory + compression), and UC Berkeley (architecture + distributed systems).

---

## Prerequisites

- Linear algebra (eigenvalues, SVD, matrix norms, positive definiteness)
- Probability and statistics (MLE, Bayesian inference, hypothesis testing)
- Calculus and optimization (gradients, Hessians, convexity, Lagrange multipliers)
- Python fluency, PyTorch basics
- Comfort with the Unix command line and Git

---

## Unit 1: Foundations of Language Modeling (Weeks 1-2)

### Week 1: Statistical Language Models and Information Theory

**Topics**
- Language modeling as next-token prediction
- Cross-entropy loss, perplexity, and bits-per-byte (BPB)
- Shannon entropy, source coding theorem, and the connection between compression and prediction
- KL divergence, mutual information
- Why BPB is the right metric for tokenizer-agnostic evaluation

**Readings**
- Shannon, "A Mathematical Theory of Communication" (1948), Sections I-III
- Jurafsky & Martin, *Speech and Language Processing*, Ch. 3 (N-grams and Language Models)
- MacKay, *Information Theory, Inference and Learning Algorithms*, Ch. 1-6

**Exercises**
- Implement a character-level n-gram model and compute BPB on a text corpus
- Derive the relationship between cross-entropy loss, perplexity, and bits-per-byte
- Prove that cross-entropy is minimized when the model distribution equals the true distribution

### Week 2: The Transformer Architecture

**Topics**
- Self-attention: queries, keys, values, scaled dot-product attention
- Multi-head attention and why it works (subspace decomposition)
- Position encodings: sinusoidal, learned, RoPE
- Layer normalization variants: LayerNorm, RMSNorm, pre-norm vs post-norm
- Feed-forward networks: expansion factor, activation functions (ReLU, GELU, SwiGLU, ReLU^2)
- Residual connections and signal propagation in deep networks
- Autoregressive generation and causal masking

**Readings**
- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
- Shazeer, "GLU Variants Improve Transformer" (2020)

**Exercises**
- Implement a transformer decoder from scratch in PyTorch (no nn.Transformer)
- Implement RoPE and verify it produces correct relative position attention patterns
- Train a small (1-layer, 64-dim) language model on a toy corpus and verify convergence

---

## Unit 2: Scaling Laws and the Parameter Golf Objective (Week 3)

### Week 3: Neural Scaling Laws and L(N) Optimization

**Topics**
- Kaplan et al. scaling laws: L(N), L(D), L(C) and their power-law relationships
- Chinchilla optimal: compute-optimal allocation between parameters and data
- The Parameter Golf objective as L(N) optimization: minimize loss given fixed N, unconstrained D and C
- Implications: at fixed N, how do depth, width, and architecture affect loss?
- Depth vs width tradeoffs: why deeper is more parameter-efficient (to a point)
- The role of the 10-minute training constraint as a soft compute bound

**Readings**
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- Tay et al., "Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers" (2022)

**Exercises**
- Fit power-law curves to training runs at 3-4 different model sizes and predict loss at a target size
- Experimentally determine: at 16MB, is it better to have 6 layers at 768d or 12 layers at 512d?
- Read the Parameter Golf leaderboard and categorize each submission's primary contribution axis (quantization, architecture, training, evaluation)

---

## Unit 3: Efficient Architectures (Weeks 4-6)

### Week 4: Grouped Query Attention, Multi-Query Attention, and KV-Cache Efficiency

**Topics**
- Multi-query attention (Shazeer 2019): shared K/V heads
- Grouped query attention (GQA): interpolating between MHA and MQA
- Parameter cost analysis: how GQA reduces attention parameters
- Flash Attention: tiling, memory-efficient backward, IO complexity
- Flash Attention 2 and 3: Hopper-specific optimizations

**Readings**
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024)

**Exercises**
- Implement GQA from scratch and verify it matches standard MHA output when num_kv_heads = num_heads
- Profile memory usage of standard attention vs Flash Attention at sequence length 2048
- Calculate exact parameter savings from GQA at various head ratios for the 512d/8-head baseline

### Week 5: Parameter-Efficient Architecture Variants

**Topics**
- Depth recurrence and the Universal Transformer (Dehghani et al.)
  - Weight sharing across layers
  - Per-layer conditioning: layer embeddings, adaptive computation time (ACT)
  - Training stability with deep recurrence (gradient flow, normalization)
- Mixture of Experts (MoE)
  - Sparse routing: top-k gating, expert capacity
  - Parameter count vs active parameter count
  - Load balancing losses
- Low-rank factorization
  - LoRA and its variants
  - Kronecker-factored layers
- U-Net / encoder-decoder skip connections in transformers
  - The skip connection pattern used in the Parameter Golf baseline
  - Learned skip weights

**Readings**
- Dehghani et al., "Universal Transformers" (2019)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (2022)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Bao et al., "All Are Worth Words: A ViT Backbone for Diffusion Models" (U-Net skips in transformers, 2023)

**Exercises**
- Implement a weight-shared transformer (3 blocks x 4 iterations) with layer index conditioning
- Compare parameter count and loss: 12-layer standard vs 3-block x 4-iteration recurrent at same width
- Implement and ablate different per-layer conditioning strategies: additive embedding, multiplicative gate, FiLM conditioning

### Week 6: Specialized Modules for Small Models

**Topics**
- BigramHash embeddings: hash-based n-gram features for small vocabularies
  - Hash function design, collision analysis
  - Learned scaling and projection
- SmearGate: temporal smoothing via per-dimension gating
- Value Embeddings (VE): re-injecting token identity at deep layers
- Cross-Sequence Attention (XSA): removing self-value projection
  - Why this forces meaningful contextual attention
  - GQA-aware implementation
- Partial RoPE: applying position encoding to a subset of dimensions
- Logit soft-capping: tanh-based logit range control

**Readings**
- Parameter Golf PRs: #162 (BigramHash), #65 (SmearGate), #374 (VE128), #478 (XSA), #315 (Partial RoPE)
- Bai et al., "Transformers as Algorithms: Generalization and Stability in In-context Learning" (2023) - theoretical grounding for why removing self-attention improves generalization

**Exercises**
- Implement BigramHash with configurable vocabulary size and embedding dimension
- Ablation study: add/remove each module (SmearGate, VE, XSA, Partial RoPE) individually and measure BPB delta
- Analyze the collision rate of BigramHash at different vocabulary sizes (1024, 2048, 3072, 4096)

---

## Unit 4: Tokenization (Week 7)

### Week 7: Tokenizers and Their Impact on BPB

**Topics**
- Byte Pair Encoding (BPE): algorithm, vocabulary construction, merge rules
- SentencePiece: unigram model vs BPE mode
- The tokenizer-agnostic BPB metric: how token-level loss converts to byte-level compression
- Why vocabulary size matters in parameter-constrained settings
  - Embedding table cost: 2 x vocab_size x model_dim (with tied embeddings: 1x)
  - Small vocab (1024): cheaper embeddings, more tokens per document, longer sequences needed
  - Large vocab (8192+): expensive embeddings, fewer tokens, each token carries more information
- Tied vs untied embeddings: parameter cost analysis
- The BigramHash approach as a middle ground: small vocab + learned n-gram features
- Byte-level tokenization and its tradeoffs

**Readings**
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2016)
- Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" (2018)
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer" (2018)

**Exercises**
- Train BPE tokenizers at vocab sizes 512, 1024, 2048, 4096, 8192 on FineWeb
- For each: compute tokens-per-byte ratio and estimate the embedding parameter cost at 512d
- Find the vocab size that minimizes total model BPB under a 16MB budget constraint

---

## Unit 5: Optimization (Weeks 8-10)

### Week 8: Optimizers for Small Model Training

**Topics**
- Adam and AdamW: momentum, adaptive learning rates, weight decay
- The Muon optimizer
  - Newton-Schulz orthogonalization: what it does and why
  - The zeropower_via_newtonschulz5 iteration: deriving the (a, b, c) coefficients
  - Muon as "spectral steepest descent" for matrix parameters
  - Momentum in Muon: Nesterov momentum, warmup
- Optimizer partitioning: different optimizers for different parameter types
  - Matrix params (Muon), scalar params (Adam), embeddings (Adam with higher LR)
- Learning rate schedules
  - Linear warmup
  - Cosine decay, linear decay
  - Wallclock-aware warmdown: adapting to variable step times
- Gradient clipping: global norm clipping, when and why

**Readings**
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2019)
- Jordan, "Muon: An optimizer for hidden layers in neural networks" (2024), blog post + code
- Bernstein et al., "Old Optimizer, New Norm: An Anthology" (2024)

**Exercises**
- Implement Muon from scratch, including the Newton-Schulz iteration
- Compare Adam vs Muon on the Parameter Golf baseline: plot train loss curves over 1000 steps
- Implement wallclock-aware warmdown and verify it adapts correctly when step times vary

### Week 9: Distributed Training and Parallel Optimization

**Topics**
- Data parallelism: gradient averaging across GPUs
- DistributedDataParallel (DDP) in PyTorch
- Gradient accumulation: simulating larger batch sizes
- All-reduce, reduce-scatter, all-gather: collective communication primitives
- NVLink and the communication topology of 8xH100 SXM
- The Parallel Muon strategy
  - Parameter Banking: storing weights as 3D tensors for batched operations
  - Overlapping communication with computation
  - Async reduce-scatter during Adam steps, then Newton-Schulz on shards
- Scaling batch size: critical batch size, gradient noise scale

**Readings**
- Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (2020)
- McCandlish et al., "An Empirical Model of Large-Batch Training" (2018)
- NVIDIA, "NCCL Developer Guide" (collective operations reference)

**Exercises**
- Profile a training step on 1 GPU vs 8 GPUs: measure communication overhead
- Implement the Parameter Banking pattern: reshape layer weights into 3D bank tensors
- Implement async reduce-scatter + all-gather with overlap and measure throughput improvement

### Week 10: Weight Averaging and Ensemble Methods

**Topics**
- Exponential Moving Average (EMA)
  - Decay rate selection, Polyak averaging
  - Why EMA helps: smoothing over loss surface noise
- Stochastic Weight Averaging (SWA)
  - Collecting snapshots during late training
  - SWA vs EMA: when each works better
- LAWA (Latest-k Weight Average)
- Combining EMA and SWA
- Connection to flat minima and generalization
- When to start averaging: the lr_scale threshold approach

**Readings**
- Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992)
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (SWA, 2018)
- Kaddour et al., "Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging" (LAWA, 2022)

**Exercises**
- Implement EMA with configurable decay and compare final BPB: EMA vs no EMA
- Implement SWA triggered by lr_scale threshold and measure improvement
- Experiment: what EMA decay rate is optimal for the Parameter Golf training duration?

---

## Unit 6: Quantization and Compression (Weeks 11-13)

### Week 11: Post-Training Quantization Fundamentals

**Topics**
- Fixed-point number representation: int8, int6, int5, int4
- Per-tensor vs per-row vs per-channel quantization
- Symmetric vs asymmetric quantization
- Calibration: choosing scale and zero-point
  - Min-max, percentile clipping, MSE-optimal
- The quantization error budget: how rounding errors accumulate through layers
- Round-to-nearest vs more sophisticated rounding

**Readings**
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
- Nagel et al., "A White Paper on Neural Network Quantization" (Qualcomm, 2021)
- Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021)

**Exercises**
- Implement per-row int8 quantization with percentile clipping (reproduce the baseline's approach)
- Measure BPB degradation from int8, int6, int5, int4 uniform quantization on the trained baseline
- Plot reconstruction MSE vs bits-per-weight for different clipping percentiles

### Week 12: Advanced Quantization (GPTQ, QAT)

**Topics**
- GPTQ: Hessian-informed quantization
  - The optimal brain quantization (OBQ) framework
  - Hessian collection: H = X^T X from calibration data
  - Cholesky decomposition and error compensation
  - Column reordering for better error propagation
  - Block-wise quantization for efficiency
  - GPTQ-lite: diagonal Hessian approximation
  - Full Hessian GPTQ: the complete algorithm
- Calibration data strategies
  - Training data calibration (ruled out by Parameter Golf rules during eval)
  - Autoregressive self-generated calibration: the model generates its own data
- Quantization-Aware Training (QAT)
  - Straight-Through Estimator (STE): gradients through discontinuous rounding
  - Fake quantization during training
  - Late QAT: enabling STE only in the final training phase (when lr is low)
  - Why late QAT works: the model learns to place weights near quantization grid points
- Mixed precision: different bit-widths for different layers or tensor types

**Readings**
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2023)
- Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization" (AdaRound, 2020)
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" (STE, 2013)

**Exercises**
- Implement GPTQ with diagonal Hessian (GPTQ-lite) and compare vs uniform quantization
- Implement full Hessian GPTQ with Cholesky error compensation
- Implement late QAT with STE: add fake quantization to CastedLinear when lr_scale < threshold
- Implement AR self-generated calibration: generate sequences from the trained model for GPTQ

### Week 13: Compression and Artifact Size Optimization

**Topics**
- Entropy coding: Huffman, arithmetic coding, ANS
- General-purpose compression: zlib, zstd, lzma
  - Why lzma compresses quantized weights better than zlib
  - Compression level tradeoffs (speed vs ratio)
- The 16MB artifact budget: code bytes + compressed model bytes
  - Strategies for minimizing code size
  - Strategies for maximizing compressibility of quantized weights
- Selective pruning: removing low-impact quantized values
  - Reconstruction error as pruning criterion
  - Binary search for size target
- Ternary quantization: {-1, 0, +1} weights
  - Extreme compression, extreme loss of precision
  - When it can work (with enough parameters and proper scaling)
- 1-bit quantization: binary weights with learned scales

**Readings**
- Zhu et al., "Trained Ternary Quantization" (2017)
- Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks" (2016)
- The LZMA SDK documentation and compression algorithm description

**Exercises**
- Compare zlib-9, zstd-22, and lzma-9 compression ratios on the same quantized model checkpoint
- Implement selective pruning: sort quantized values by reconstruction error, prune to hit size target
- Compute: at int6 with lzma-9, how many raw parameters can fit in 16MB? What about int4?

---

## Unit 7: Evaluation Methods (Week 14)

### Week 14: Evaluation Strategies and Test-Time Compute

**Topics**
- Standard autoregressive evaluation: fixed context, sequential BPB
- Sliding window evaluation
  - Stride selection: trading compute for better context
  - Why it improves BPB: each token sees maximum available context
  - Implementation: scoring only "new" tokens to avoid double-counting
- Test-Time Training (TTT)
  - Adapting model parameters on previously-evaluated tokens
  - LoRA TTT: lightweight adaptation during evaluation
  - Legal TTT in Parameter Golf: only train on tokens you've already scored
  - Score-first TTT: evaluate, then adapt, ensuring no data leakage
- Long-context evaluation
  - Extending eval sequence length beyond training length
  - Position extrapolation: NTK-aware RoPE scaling, YaRN
  - Memory and compute costs of long-context eval
- Evaluation time budget: fitting within 10 minutes on 8xH100

**Readings**
- Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization" (ALiBi, 2022)
- Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (TTT, 2024)
- Bloc97, "NTK-Aware Scaled RoPE" (2023, online post)

**Exercises**
- Implement sliding window evaluation with configurable stride
- Measure BPB improvement from sliding window eval at strides 32, 64, 128, 256
- Implement a minimal LoRA TTT loop: fine-tune rank-4 LoRA on evaluated tokens, measure BPB delta
- Profile eval time: how many sliding window passes fit in 10 minutes on 1xH100?

---

## Unit 8: Systems and Performance (Week 15)

### Week 15: GPU Programming, Kernels, and Training Throughput

**Topics**
- GPU architecture: SMs, warps, memory hierarchy (registers, shared memory, L2, HBM)
- The H100 Hopper architecture: TMA, warp specialization, FP8 tensor cores
- torch.compile: how it works, tracing, graph breaks, fullgraph mode
- Profiling training runs: torch.profiler, nsight systems, identifying bottlenecks
- Memory optimization
  - Activation checkpointing / gradient checkpointing
  - Mixed precision training: bf16 forward, fp32 optimizer state
  - Memory-efficient attention (Flash Attention) vs standard attention memory cost
- Maximizing tokens per second
  - Batch size tuning for GPU utilization
  - Reducing Python overhead: compiled models, fused kernels
  - Data loading: async prefetch, pinned memory, non-blocking transfers
- Custom CUDA kernels and Triton: when and why
  - Fused operations (e.g., fused RMSNorm + linear)
  - Megakernels: combining multiple operations into a single GPU launch

**Readings**
- NVIDIA, "H100 Tensor Core GPU Architecture" whitepaper (2022)
- Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (2024)
- Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2019)

**Exercises**
- Profile a training step with torch.profiler: identify the top 5 time-consuming operations
- Implement activation checkpointing for the transformer blocks and measure memory savings
- Write a Triton kernel for fused RMSNorm and benchmark against PyTorch's F.rms_norm
- Experiment: what is the maximum batch size (tokens/step) that fits in 80GB H100 HBM for the baseline model?

---

## Unit 9: Integration and Competition Strategy (Week 16)

### Week 16: Putting It All Together

**Topics**
- The SOTA stack walkthrough: reading and understanding the current #1 submission line by line
- Contribution axes and diminishing returns analysis
  - Where is the most remaining headroom?
  - Quantization frontier: int6 to int5 to int4, marginal gains
  - Architectural innovations: what hasn't been tried?
  - Training efficiency: can we fit more steps in 600 seconds?
  - Evaluation tricks: how much more can sliding window + TTT give?
- Experiment design for competition
  - Ablation methodology: single-variable changes, controlled comparisons
  - Statistical significance: Welch's t-test, p < 0.01, 3-seed validation
  - The 0.005-nat improvement threshold
- Submission process and reproducibility
  - Records folder structure
  - Logging and artifact generation
  - README documentation standards
- Risk management
  - Compute cost estimation before launching 8xH100 runs
  - Iterating cheaply on 1xH100 or local GPU before scaling
  - Version control for experiments: branches, tags, logs

**Readings**
- The full PR history of Parameter Golf: #1019, #549, #414, #287, #198, #162, #65
- All README files in the records/track_10min_16mb/ directory
- The Parameter Golf FAQ and rules (challenge page + README)

**Capstone Project**
- Starting from the current SOTA codebase, implement one novel improvement
- Run 3-seed validation on 8xH100
- Prepare a complete submission: README, submission.json, train_gpt.py, train logs
- Present results with statistical analysis (mean, std, Welch's t-test)

---

## Appendix A: Key Papers Reference List

| Topic | Paper | Year |
|-------|-------|------|
| Transformer | Vaswani et al., "Attention Is All You Need" | 2017 |
| Scaling Laws | Kaplan et al., "Scaling Laws for Neural Language Models" | 2020 |
| Chinchilla | Hoffmann et al., "Training Compute-Optimal Large Language Models" | 2022 |
| RoPE | Su et al., "RoFormer" | 2021 |
| Flash Attention | Dao et al., "FlashAttention" | 2022 |
| Universal Transformer | Dehghani et al., "Universal Transformers" | 2019 |
| BPE | Sennrich et al., "Neural Machine Translation of Rare Words" | 2016 |
| Adam | Kingma & Ba, "Adam" | 2015 |
| SWA | Izmailov et al., "Averaging Weights Leads to Wider Optima" | 2018 |
| GPTQ | Frantar et al., "GPTQ" | 2023 |
| STE | Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" | 2013 |
| MoE | Fedus et al., "Switch Transformers" | 2022 |
| LoRA | Hu et al., "LoRA" | 2021 |
| TTT | Sun et al., "Learning to (Learn at Test Time)" | 2024 |

## Appendix B: Recommended Tool Proficiency

| Tool | Purpose | Priority |
|------|---------|----------|
| PyTorch | Model implementation, training loops | Critical |
| torch.compile | Kernel fusion, graph optimization | Critical |
| Flash Attention | Efficient attention computation | Critical |
| torchrun | Distributed training launcher | Critical |
| NCCL | GPU-to-GPU communication | High |
| SentencePiece | Tokenizer training and inference | High |
| torch.profiler | Performance analysis | High |
| nsight systems | GPU-level profiling | Medium |
| Triton | Custom GPU kernels | Medium |
| CUDA | Low-level GPU programming | Medium |
| lzma/zstd/zlib | Model compression | Medium |

## Appendix C: Compute Planning

| Phase | Hardware | Estimated Hours | Cost Estimate |
|-------|----------|-----------------|---------------|
| Weeks 1-6 exercises | Local GPU or 1xH100 | 20 hrs | $50 |
| Weeks 7-10 exercises | 1xH100 | 30 hrs | $75 |
| Weeks 11-13 exercises | 1xH100 | 20 hrs | $50 |
| Week 14 eval experiments | 1-8xH100 | 15 hrs | $75 |
| Week 15 profiling | 1xH100 | 10 hrs | $25 |
| Week 16 capstone | 8xH100 | 20 hrs | $400 |
| **Total** | | **~115 hrs** | **~$675** |
