# Lim Shiaw Yong - 12MB Squeeze (QAT6 + SwiGLU)

## Performance Metrics
* **Validation BPB:** 1.6644
* **Final Payload Size:** 11.74 MB
* **Hardware:** NVIDIA RTX A6000 (Simulated for 8xH100 CI compatibility)

## Architectural Modifications
To aggressively crush the file size while maintaining gradient flow, this submission guts the baseline transformer and implements the following core optimizations:

### 1. Symmetrical Modulo Routing (Depth Recurrence)
The network physically instantiates only 6 unique Transformer blocks to save VRAM and file size. However, data is routed through these blocks using a palindrome loop, simulating a 12-layer logical depth (e.g., `0→1→2→3→4→5→5→4→3→2→1→0`). 

### 2. Parallel Residuals
To prevent the gradients from vanishing during the Deep Recurrence loops, the Attention and MLP blocks are calculated simultaneously and injected back into the residual stream at the exact same time, widening the backward pass.

### 3. TTT Micro-Batching
Standard Test-Time Training (TTT) was bottlenecked by evaluating 500k-token batches before taking a single gradient step. The `eval_val` loop has been rewritten to subdivide the evaluation chunks, allowing the isolated SGD optimizer to take rapid, continuous micro-steps during the test.

### 4. The Entropy Squeeze (Int6 QAT)
The model was trained using Quantization-Aware Training at `QAT_BITS=6`. This 6-bit quantization noise was injected during the active training loop, allowing the network to route its logic around the compression. This resulted in the final zlib payload compressing down to 11.7MB, leaving over 4MB of headroom under the 16MB ceiling.

## Execution Schedule
The model was prototyped using a fast-burn micro-batching schedule to verify structural integrity:
* `TRAIN_BATCH_TOKENS`: 65,536
* `ITERATIONS`: 1,200
* `WARMUP_STEPS`: 20