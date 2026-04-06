# SOTA Component Map

Reference: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
(2135 lines, 101,850 bytes)

## Quick Reference: Component -> Lines -> Env Vars

| Component | Lines | Key Env Vars | Notes |
|-----------|-------|-------------|-------|
| Hyperparameters | 28-96 | See full list below | All config via os.environ |
| Muon Optimizer | 98-262 | MATRIX_LR, MUON_* | Newton-Schulz orthogonalization |
| Control tensor patterns | 359-374 | CONTROL_TENSOR_NAME_PATTERNS | Determines which params go to Adam vs Muon |
| CastedLinear + Late QAT | 537-549 | QAT_ENABLED, LATE_QAT_THRESHOLD | STE fake-quant in forward when enabled |
| Rotary (Partial RoPE) | 555-596 | ROPE_DIMS, ROPE_BASE | Only first rope_dims of each head get RoPE |
| CausalSelfAttention | 598-668 | NUM_HEADS, NUM_KV_HEADS | GQA + FA3 + optional XSA |
| XSA (Extended Self-Attention) | 633-642 | XSA_LAST_N | Orthogonal value subtraction post-attention |
| SmearGate | 671-678 | (none, always active) | Sigmoid gate mixing current + shifted positions |
| BigramHash | 680-712 | BIGRAM_VOCAB_SIZE, BIGRAM_DIM, TRIGRAM | Hash-based ngram embedding |
| VE128 (Value Embedding) | 714-729 | VE_ENABLED, VE_DIM, VE_LAYERS | Shared token->value embedding for deep layers |
| MLP (LeakyReLU^2) | 731-737 | MLP_MULT | leaky_relu(x, 0.5) then square |
| Block | 739-793 | LN_SCALE | RMSNorm + attention + MLP + residual scales |
| GPT model | 795-1000 | NUM_LAYERS, MODEL_DIM, TIE_EMBEDDINGS | Encoder-decoder U-Net with parameter banks |
| Parameter Banks | 826-834, 887-897 | (derived from model shape) | qo/kv/mlp_up/mlp_down stacked as 3D tensors |
| U-Net Skips | 822-825, 923-937 | NUM_LAYERS | First half encodes, second half decodes with skip |
| Sliding Window Eval | 1010-1078 | EVAL_STRIDE, EVAL_SEQ_LEN | Stride-based overlapping eval for better BPB |
| AR Calibration Gen | 1081-1101 | SEED, TRAIN_SEQ_LEN | 64 seqs x 2048 tokens, temp=0.8 |
| Hessian Collection | 1104-1137 | (none) | H = X^T X from AR-generated activations |
| GPTQ Quantizer | 1171-1224 | GPTQ_BLOCK_SIZE (default 128) | Full Hessian Cholesky + column reorder |
| Percentile int6 (fallback) | 1226-1245 | (none) | For tensors without Hessians |
| Unbank/Rebank | 1247-1312 | (none) | Convert banked params for GPTQ/eval |
| Hessian Model (_HessianGPT) | 1313-1491 | (none) | Duplicate model that collects H = X^T X |
| Mixed Quantize int6 | 1493-1527 | (none) | GPTQ for attn/mlp weights, percentile for rest |
| Dequantize | 1528-1547 | (none) | int6 -> float32 reconstruction |
| Training Loop | 1811-1913 | ITERATIONS, MAX_WALLCLOCK_SECONDS | Core train with SWA/EMA/LAWA updates |
| EMA (decay 0.997) | 1880-1883 | (hardcoded) | Updated every step |
| SWA | 1886-1894 | SWA_ENABLED, SWA_EVERY | Snapshots when LR scale < 0.2 |
| Post-train: EMA/LAWA load | 1917-1933 | LAWA_ENABLED, LAWA_K | Chooses averaged weights |
| AR Calib + GPTQ pipeline | 1979-1994 | TARGET_MB | Generate calib -> collect H -> quantize |
| Selective +-1 Pruning | 1995-2041 | TARGET_MB | Zero small quant values if it reduces error |
| LZMA Compression | 2042-2052 | (preset=9 hardcoded) | Final artifact compression |
| Sliding Window Final Eval | 2099-2130 | EVAL_STRIDE | The score that matters (BPB with stride=64) |

## Key Hyperparameter Defaults (from Hyperparameters class)

| Param | Default | SOTA Override (from run command) |
|-------|---------|----------------------------------|
| num_layers | 11 | (default) |
| model_dim | 512 | (default) |
| num_heads | 8 | (default) |
| num_kv_heads | 4 | (default) |
| mlp_mult | 3 | (default) |
| vocab_size | 1024 | (default) |
| train_seq_len | 2048 | (default) |
| train_batch_tokens | 786432 | (default) |
| warmdown_iters | 4000 | WARMDOWN_ITERS=4000 |
| bigram_vocab_size | 3072 | BIGRAM_VOCAB_SIZE=3072 |
| bigram_dim | 112 | BIGRAM_DIM=112 |
| xsa_last_n | 11 | (default = all layers) |
| rope_dims | 16 | (default) |
| ln_scale | True | (default) |
| ve_enabled | True | (default) |
| ve_dim | 128 | (default) |
| ve_layers | "9,10" | (default) |
| late_qat_threshold | 0.15 | (default) |
| swa_every | 50 | (default) |
| max_wallclock_seconds | 600.0 | (default) |
| seed | 314 | SEED=314 |
| target_mb | 15.9 | TARGET_MB=15.9 |

## Training Flow (annotated from seed=314 log)

```
Step     Event                        BPB/Loss
0        Initial val                  4.1026 BPB
1-10     Early training               loss 6.93 -> 6.05
500      Training                     loss 2.38
4000     Mid-training val             1.2051 BPB
6150     SWA starts                   (LR scale < 0.2)
6335     Late QAT enabled             (LR scale < 0.15)
6927     Final val (pre-quant)        1.1354 BPB
6927     Wall clock cap hit           600,109 ms
---      EMA weights applied          1.1344 BPB (diagnostic)
---      AR calib generation          64 seqs, 196.7s
---      Hessian collection           68 layers
---      Selective pruning            no pruning needed (15.13 MB)
---      int6+LZMA serialize          15,761,428 bytes model
---      Total artifact               15,863,278 bytes
---      int6 roundtrip eval          1.1386 BPB (no sliding)
---      Sliding window eval          1.1151 BPB (stride=64) <- THIS IS THE SCORE
```
