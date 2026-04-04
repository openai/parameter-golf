# 0.4188 BPB - Mixed Precision Quantization + Causal Backoff N-gram + Complementary Training

**Author:** Lucas Ercolano  
**Track:** 16MB / 10-minute 8xH100 SXM  

This submission drastically reduces the BPB by combining a highly optimized neural baseline with a strict, DDP-safe causal n-gram mixer and complementary training, fitted into the 16MB artifact limit via asymmetric mixed-precision quantization.

## Results (3 Seeds)

* **Seed 42:** 0.418682 BPB
* **Seed 1337:** 0.416106 BPB
* **Seed 7:** 0.421661 BPB
* **Average:** **0.418816 BPB**

*All runs comfortably complete within the 600s time limit and comply with the 16MB artifact constraint (Max artifact size: 15.66 MB).*

## Key Innovations & Architecture

### 1. Mixed Precision Quantization (Int5 / Int6)
To fit a highly capable 27M parameter model within the 16MB limit while retaining high entropy weights, the model applies asymmetric quantization during the artifact compression phase:
* **MLP Layers:** Quantized to `int5`.
* **Attention / Embeddings:** Quantized to `int6`.
* **Dynamic QAT:** The `CastedLinear` modules simulate Quantization-Aware Training with dynamic clipping values based on the target layer, preventing parameter degradation during the final compression to LZMA.

### 2. Complementary Training
The neural model is explicitly trained to specialize in tokens that are hard for n-grams to predict. The loss is re-weighted down for tokens easily predicted by bigram statistics: `w_i = 1 - alpha * p_bigram(token_i)`.

### 3. Strictly Legal Causal Backoff N-gram Mixer
The evaluation loop utilizes a Backoff N-gram Mixer with entropy-adaptive alpha blending. 
* **Score-First Legality:** The mixer updates only *after* the tokens have been formally evaluated by the sliding window. 
* **DDP Synchronization:** To prevent causal leaks across the multi-GPU setup, the mixer enforces a strict `dist.barrier()` before updating its internal state. This ensures no GPU injects future tokens into the cache before all instances have finished scoring the current chunk.

### Base Neural Stack (Derived from PR #549)
* 11L GQA Transformer, 512d, 8 heads, 4 KV heads.
* MLP 3.0x with LeakyReLU(0.75)².
* Parallel Muon optimizer.
* SmearGate + BigramHash(2048) + OrthoInit.
* Value-Residual Embeddings (VE128).

## Artifact Pipeline
The submission relies on `lzma` for compressing the mixed-precision state dictionary, achieving a final payload size of ~15.6 MB (including the `train_gpt.py` script).
