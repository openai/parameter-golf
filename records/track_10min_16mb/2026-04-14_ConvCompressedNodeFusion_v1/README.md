The key idea is using residual approximation to generate the LLM output. Parameter reduction is 
obtained by using lower dimensional attention in the residual nodes. 

Each residual node first compresses features from d_model to a smaller d_attn space via depthwise + 
pointwise 1D convolution (EmbeddingSemanticConv). Attention is then computed only in this compressed space 
(CausalCompressedAttention) and projected back to model width. This keeps transformer-style sequence modeling 
while reducing attention parameters and compute versus full-width attention throughout.

Each block contains multiple heterogeneous nodes (different conv kernel/dilation patterns), fused by learned
softmax scalar weights (NodeFusion). This behaves like a lightweight internal ensemble with minimal extra

Parameter count is further reduced by tying output logits to token embeddings (no separate LM head matrix),
using mostly bias-free linear/conv layers, and applying RMSNorm.

Training remains simple and stable: next-token cross-entropy, AdamW, cosine LR with warmup, gradient 
accumulation, mixed precision (bf16/fp16 autocast), optional torch.compile, and DDP support. The loop is
wall-clock aware (time_limit_minutes) and resume-capable (resume_path, optional step reset/additional steps), 
making it practical for strict runtime budgets.

CORE STRUCTURE:
  1. Token embedding + learned positional embedding.
  2. 3 blocks (n_blocks=3) of NodeFusion.
  3. Each block has 4 approximators (n_nodes=4), where each approximator is a ResidualApproxNode.
  4. Each block is run with 2 recursions (recurrence=2).
  5. Final RMSNorm + tied-output projection (uses embedding weights as LM head).

Each approximator contains:
  - RMSNorm
  - Conv compressor (d_model -> d_attn, depthwise + pointwise)
  - Causal compressed attention
  - Linear expand (d_attn -> d_model)
  - FFN (384 -> 512 -> 384) with SiLU and residuals.

