# Parameter Golf: Ideas

## Autoresearch Sweep Queue
- **Architecture grid sweeps (active)**: Sweep `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, and `MLP_MULT` on small Modal runs, rank by best `val_bpb`, and only keep changes that still respect the 16,000,000-byte artifact cap.
- **Training schedule sweeps (next)**: Sweep `WARMUP_STEPS`, `WARMDOWN_ITERS`, and the optimizer LRs once the architecture grid is stable.
- **PR #250-inspired low-risk ideas**: Test SGDR/cosine restarts, then only if needed consider simple transferable pieces like lower-cost routing-inspired capacity changes or cheaper attention settings before attempting full MoE/PID complexity.
- **Tokenizer family sweeps (later)**: Compare `sp1024` against larger vocabulary variants only after the sweep machinery is stable and size accounting is explicit.

## Quantization & Compression
- **INT4 Quantization**: Double the effective parameter budget from ~19M to ~38M by moving from int8 to int4 weight quantization. The single highest-leverage change available — requires careful per-channel scaling to limit accuracy loss.
- **Mixed-Precision Quantization (INT4/INT8)**: Keep embedding and attention projections in int8 while quantizing MLP weights to int4. Balances compression ratio with sensitivity of different layer types.
- **GPTQ / AWQ-style Quantization**: Use calibration data to find optimal rounding decisions during quantization. Can recover 0.5-1% of the naive quantization loss at no size cost.
- **Learned Quantization Scales**: Train quantization step sizes jointly with model weights (QAT). Eliminates the post-training quantization gap entirely at the cost of slower training.
- **Pruning + Quantization Pipeline**: Structured pruning to remove entire attention heads or MLP neurons before quantizing. Compound compression — prune 20% then int4 quantize for ~2.5x effective parameter increase.

## Vocabulary & Tokenization
- **Optimal Vocab Size (2048-4096)**: Increase vocab from 1024 to reduce tokens-per-byte by 15-30%, directly lowering BPB. Sweet spot balances embedding table cost against sequence compression (see `research/optimal_vocab_size.md`).
- **Variable-Dimension Embeddings (Matryoshka)**: Assign higher-dimensional embeddings to frequent tokens and lower-dimensional to rare ones, projecting up to model_dim. Exploits the Zipfian distribution of language — top 200 tokens cover ~80% of text.
- **Byte-Fallback Tokenization**: Use a smaller core vocab with byte-level fallback for OOV, avoiding wasted embedding rows. Keeps the embedding table lean while maintaining full coverage.

## Architecture
- **Depth Recurrence (Universal Transformer)**: Reuse the same block weights across multiple layers, paying parameter cost once but getting depth for free. A 9-layer model with 3x weight sharing acts like 27 layers at 9-layer parameter cost.
- **Mixture of Experts (MoE)**: Replace dense MLP with 4-8 small experts and a learned router, activating only 1-2 per token. Gets 2-4x effective MLP capacity with minimal parameter overhead from the router.
- **Paired-Head Attention on Steroids**: Go beyond GQA — pair query heads that share not just KV but also learned relative mixing coefficients. Cuts KV projection parameters further while maintaining representational diversity through learned head interactions.
- **Manifold Ultra-Connections**: Replace linear skip connections with learned low-rank nonlinear transforms between encoder and decoder halves. Richer information flow across the U-Net skip topology at negligible parameter cost (~rank 16-32 bottleneck).
- **Hyper-Efficient Attention (Linear / Performer)**: Replace softmax attention with a linear approximation for some layers. Frees up FLOP budget to run more training steps within the 10-minute wall clock.
- **Sub-Quadratic Feedforward**: Replace relu^2 MLP with a sparse lookup or product-key memory. Same expressivity at lower parameter count — each token retrieves a small subset of a large implicit weight matrix.

## Training & Optimization
- **FP8 Training (H100)**: Use FP8 matmuls for forward/backward passes to nearly double throughput. More training steps in 10 minutes = lower loss at the same parameter count.
- **Progressive Growing**: Start training with fewer layers and smaller sequences, then grow. Faster early iterations let the model see more data in the same wall clock budget.
- **Aggressive Learning Rate Schedules**: Use WSD (warmup-stable-decay) with a much shorter stable phase. Matched to the 20K iteration budget, this can squeeze out a few percent lower loss.
- **Distillation from a Larger Run**: Train a 40M+ parameter teacher unconstrained, then distill into the submission model. The student can learn softer targets that compress better than raw data.

## Radical / Speculative
- **Sparse Circuit Discovery & Compression During Training**: Identify and freeze critical computational circuits mid-training, then prune everything else. Combines lottery ticket hypothesis with online structure discovery — train once, compress by finding the winning subnetwork automatically.
- **Decision Tree Distillation**: Distill the final language model into a hybrid architecture mixing neural layers with learned decision trees for frequent patterns. Tree components compress to near-zero size and handle the long tail of predictable n-gram patterns perfectly.
- **Neural Architecture Search (NAS) within Budget**: Use a supernetwork with weight sharing to search over depth, width, head count, and MLP ratio jointly. Finds the Pareto-optimal architecture for the 16MB constraint rather than guessing.
- **Kolmogorov Complexity-Aware Training**: Add a regularizer that penalizes weight entropy directly, encouraging maximally compressible weight distributions. Trains the model to be good AND small simultaneously rather than training then compressing.
- **Tensor Train / Low-Rank Factorization**: Decompose all weight matrices into tensor-train format with learned ranks. Can achieve 3-5x compression on MLP weights with minimal accuracy loss if ranks are tuned per layer.
- **Activation Checkpointing + Wider Model**: Trade compute for memory to train a much wider model within GPU memory, then compress. Wider models compress better than deeper ones due to more redundancy in weight matrices.
