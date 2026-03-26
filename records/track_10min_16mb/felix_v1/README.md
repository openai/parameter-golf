# felix_v1

## Changes from baseline

### 1. int6 quantization (body weights)
- Default int8 [-127, 127] replaced with int6 [-31, 31]
- Same per-row scheme, same storage (int8 container), but 6-bit value range
- zlib → zstd level-22 compression (~10-15% smaller artifact)
- Result: ~25% smaller artifact for body weights

### 2. fp16 embedding passthrough
- Tied embedding / lm_head kept at fp16 instead of quantized
- Embeddings are the most quantization-sensitive tensor (directly affects logit quality)
- Net artifact cost vs int6 embedding: ~400KB at dim=512, worth the quality gain

### 3. MLP 3x (default)
- MLP_MULT=3 instead of 2
- Space freed by int6+zstd enables wider hidden layers within 16MB budget
- Model body: ~21.4M params at dim=512, 9 layers, mlp_mult=3 → fits in ~14MB int6+zstd

### 4. seq4096 training (default)
- TRAIN_SEQ_LEN=4096 instead of 1024
- Longer context during training → model learns longer-range dependencies
- Enables sliding window eval to show real gains

### 5. Sliding window eval
- After quantized roundtrip eval, runs sliding window evaluation (stride=512)
- Each token scored with seq_len tokens of left context instead of variable (0 to seq_len)
- Free bpb improvement — no artifact cost, uses eval compute budget
- Expected gain: ~0.034 bpb on well-trained models (per PR #88)

## Expected artifact size
- Body (21.4M params, int6): ~16MB × 0.75 = 16.0MB raw → ~14.1MB zstd-22
- Embedding (512×512, fp16): 0.5MB
- Total: ~14.6MB → safely under 16MB

## Config
All hyperparameters are env-var overridable. Defaults:
- MLP_MULT=3
- TRAIN_SEQ_LEN=4096
- WEIGHT_QUANT_BITS=6
- EMBED_QUANT_BITS=0 (fp16 passthrough)
- SLIDING_WINDOW_STRIDE=512
