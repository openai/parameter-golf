# 11L + XSA + VRL + SWA + seq4096 + cross-doc TTT

**val_bpb: 1.1839** (post int8+zlib quantization roundtrip + cross-doc TTT, sliding window stride=64)
**val_bpb: 1.2192** (post int8+zlib quantization roundtrip, no TTT)

Trained on 8xH100 SXM in 600 seconds. 15.35 MB artifact (int8+zlib).

## Key Techniques

### 1. seq_len=4096 Long-Context Training
Training sequence length increased from 1024 to 4096 tokens. Provides much richer context signal during training, significantly improving sliding window eval where longer context can be exploited. Single largest contributor to improvement over the notapplica baseline.

### 2. Exclusive Self-Attention (XSA)
Applied to the deepest 4 transformer layers. After SDPA, subtracts from each head's output the component aligned with its value vector: `output -= dot(output, v_norm) * v_norm`. Forces attention outputs to carry new information rather than copying values, improving information flow. GQA-aware via reshape+broadcast. Validated at -0.008 bpb at 500 steps.

### 3. Value Residual Learning (VRL)
Each layer's value vector gets a residual from layer 0's value: `v_i += lambda_i * v_first`, where `lambda_i` is a learnable scalar initialized to 0. Provides a direct path for early token features to influence all layers. Validated at -0.014 bpb pre-quant at 500 steps.

### 4. SmearGate
Learned gate blending each token's representation with the previous token's representation. Provides lightweight bigram-level context. 512 parameters, initialized to zeros (identity at init).

### 5. Stochastic Weight Averaging (SWA)
24 checkpoints from the last 40% of warmdown averaged before export. Produces smoother weight distributions.

### 6. Cross-Document TTT (Test-Time Training)
At evaluation, rank-8 LoRA adapters are trained per document on already-evaluated tokens (no data leakage). Adapters target Q/V projections across all layers. Reset between documents.

### 7. Warmdown-QAT
warmdown_iters=1200, training entirely within warmdown phase. Reduces quantization penalty to near-zero (+0.0009 bpb, 1.2183 → 1.2192).

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 2x expansion, relu^2 activation
- XSA on layers 7-10 (deepest 4)
- VRL lambdas on layers 1-10
- SmearGate at embedding layer
- Tied embeddings (FP16 export)
- OvertoneInit + phase-transition resid_mix

## Training Hyperparameters

| Parameter | Value |
|---|---|
| num_layers | 11 |
| model_dim | 512 |
| mlp_mult | 2 |
| train_seq_len | 4096 |
| train_batch_tokens | 524,288 |
| iterations | 13137 (wallclock capped) |
| warmdown_iters | 1200 |
| matrix_lr | 0.06 |
| scalar_lr | 0.06 |
| tied_embed_lr | 0.07 |
| muon_weight_decay | 0.02 |
| eval_stride | 64 |
| swa_n_checkpoints | 24 |
| swa_frac | 0.4 |

## Results

| Metric | Value |
|---|---|
| pre-quant val_bpb | 1.2183 |
| post-quant val_bpb (int8+zlib) | 1.2192 |
| post-quant + cross-doc TTT | **1.1839** |
| model size | 15.35 MB |
| training steps | 13137 / 20000 |
| step_avg | 45.68 ms |

## Training Log

See `train_seed1337.log`.
