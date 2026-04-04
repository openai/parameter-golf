# Trying to achieve the baseline with a single H100

**Author:** Aditya Sasidhar (@adityasasidhar)

This record represents an attempt to push the 10-minute, 16MB "Parameter Golf" track to its limits using a single H100 GPU. By employing aggressive quantization and architectural optimizations, we aim to maximize model capacity while staying within the strict time and size constraints.

## Strategy

### 1. Architectural Scaling
- **11 Layers**: Increased depth from the typical 9-10 layers to 11 to improve representational capacity.
- **Grouped Query Attention (GQA)**: Using 8 query heads and 4 KV heads. This reduces the parameter count for KV projections and speeds up inference/evaluation without significantly hurting performance.
- **Partial RoPE**: Rotary Positional Embeddings are applied only to the first 16 dimensions of the head dimension, balancing positional awareness with computational efficiency.
- **XSA (Extended Attention)**: Applied on the last 4 layers.
- **MLP Expansion**: A 3x expansion factor in the MLP blocks ( despite trying to achieve the baseline on a single h100 i know )

### 2. Aggressive Quantization
To fit an 11-layer model (~24M parameters) into a 16MB submission, we use a mixed quantization approach:
- **Int6 Quantization**: Applied to the largest weight matrices, including all MLP and Attention projections (`mlp.fc`, `mlp.proj`, `attn.c_qkv`, `attn.proj`).
- **Int8 / FP16**: Used for smaller control tensors and embeddings to maintain precision where it matters most.
- **Zlib Compression**: The quantized state dict is further compressed using zlib to squeeze under the 16MB cap.

### 3. Training & Optimization
- **Single H100**: The entire training run is completed on a single H100 GPU in under 600 seconds.
- **Muon Optimizer**: We use the Muon optimizer to accelerate the convergence of the internal representation matrices, which is critical for the short 10-minute training window.
- **Late QAT (Quantization Aware Training)**: 
    - The model trains in high precision (BF16) for the first 60% of the run.
    - For the final **40% of training (QAT40)**, we enable Quantization Aware Training. This allows the model to adapt its weights to the specific rounding errors introduced by Int6/Int8 quantization, significantly recovering the validation loss hit.
- **Evaluation**: Uses sliding-window validation with a stride of 96.

## Results
- **Baseline has 3.4
- **Validation BPB**: 1.28311537 (Quantized)
- **Validation Loss**: 2.16648655 (Quantized)
- **Pre-Quantization BPB**: 1.2755
- **Wallclock Time**: 600 seconds (Hard Cap)
- **Submission Size**: 15,773,227 bytes (~15.04 MiB)

This configuration demonstrates that even with very tight size constraints, deeper models can be viable if paired with late-stage QAT and efficient attention mechanisms like GQA.

Further I would like credits so I can properly install the flash attention module to beat the baseline with a single H100 gpu, due to slower build times I ran out of credits so yeah lol.
