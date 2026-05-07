# Training on the Apple Neural Engine

## Summary

| Metric | Value |
|---|---|
| val_bpb | 3.2636 |
| Model | GolfWide: 9L, dim=512, hidden=1024, GQA 8/4, 21,767,680 params |
| Training hardware | Apple M4 Pro, Neural Engine + CPU |
| Training time | 193 seconds (wall), 164s train, 635ms compile |
| Training steps | 5,000 at 32.8 ms/step |
| Eval method | Sliding window, stride=64, seq_len=256 |

This is a **non-record submission** using the Neural Engine as the primary accelerator for transformer forward passes and selected backward computations, with CPU handling unsupported operations, weight gradients, and optimizer updates.


    
## Quick start

```bash
pip install torch numpy sentencepiece


# Verify val_bpb from the included artifact (uses sliding window stride=64 by default)
python3 train_gpt.py --eval-only \
    --load-artifact model_artifact.bin \
    --data-dir /.../fineweb10B_sp1024 \
    --tokenizer /.../fineweb_1024_bpe.model
```

Expected output: `val_bpb: 3.2636`

`--load-artifact` loads the compressed int8+zlib model, dequantizes, and runs sliding window BPB evaluation. No ANE hardware required.

## What this is

ANE-accelerated language model training on Apple Silicon. The Apple Neural Engine (ANE) is a fixed-function neural accelerator, but there is no public API for direct training on it. Apple exposes on-device model updating via Core ML and GPU-based training via Metal/MPS, but the ANE itself is restricted to inference through public APIs.

This attempt builds upon the reverse-engineered private APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`) from the **maderix** ANE project to dispatch transformer forward pass matmuls and selected backward pass computations to the ANE, with the remaining work performed on the CPU.

## Architecture

21,767,680 parameters: 21,242,880 in transformer layers, 524,288 in embeddings (tied), 512 in final RMSNorm.

| Component | Specification |
|---|---|
| Layers | 9 |
| Dimensions | dim=512, hidden=1024 |
| Attention | 8 query heads, 4 KV heads (GQA), head_dim=64 |
| FFN | SwiGLU (W1, W3 with SiLU gate, W2 projection) |
| Position encoding | RoPE (base=10000) |
| Normalization | RMSNorm |
| Residual scaling | DeepNet (alpha = 1/sqrt(2*N_layers)) |
| Embeddings | Tied (embedding = classifier) |
| Vocabulary | 1024 (sp1024 BPE) |
| Training sequence length | 256 |





### Differences from Golf baseline

Used SwiGLU instead of relu^2, Adam instead of Muon, & used DeepNet residual scaling instead of resid_mix. No logit softcapping, no QK RMSNorm, no per-head q_gain.



###  How ANE-accelerated training works & Dynamic weight pipeline
The ANE is a 16-core fixed-function accelerator. The ANE executes pre-compiled computation graphs in Apple's MIL. Weights are loaded into compiled programs. To avoid recompiling on every weight update, weights are packed w/ activations into IOSurface shared memory buffers and passed thru as input data. 10 MIL kernels are compiled once at startup, reused for all runs.

### CPU/ANE

| Operation | Device |
|---|---|
| Forward matmuls (QKV, Wo, FFN) | ANE |
| Backward activation gradients (dx) | ANE |
| Causal attention masking + softmax | CPU |
| RMSNorm forward + backward | CPU |
| SiLU derivative | CPU |
| Weight gradients (dW) | CPU (cblas_sgemm) |
| Adam optimizer | CPU |
| Loss, embedding, backward | CPU |

ANE utilization is ~2.7% due to the single sequence dispatch overhead.

### Known issues

**no causal masking in ANE SDPA.** The ANE's native scaled dot-product attention op ignores causal masks. fixed by decomposing attention on the ANE, causal mask + softmax on CPU, scores@V on ANE. ends up w/ three dispatches and two CPU roundtrips per layer.

**FP16-only compute, no loss scaling.** Backward pass gradients underflow to zero w/o manual scaling. Loss is multiplied by 256 before backprop and divided out before weight updates. At higher rates NaN appears step 13-16K. No recovery once that happens

**Single-sequence batching.** MIL kernels are compiled for `[1, DIM, 1, SEQ]` w/ no batch dimension. Each dispatch = 1 sequence of 256 tokens. Effective batch comes only from gradient accumulation on CPU

**IOSurface dispatch overhead.** Every kernel invocation requires staging inputs into IOSurface shared mem, dispatching via `_ANEClient`, and reading outputs back

**32MB SRAM cliff.** Workloads that fit in the ANE's 32MB on-chip SRAM run at peak throughput. Scaling up (hidden=1536+ or SEQ=1024 with bigger attention matrices) risks hitting this limit & moving to DRAM

## Training details

**Data:** loader detects golf's shard header and skips it.

**Hyperparameters:** lr=2e-4, warmup=500 (linear), cosine decay to 10%, accum=20, clip=0.3, Adam (beta1=0.9, beta2=0.95, eps=1e-8), wd=0.1, loss_scale=256.0, 5,000 steps.

**Effective batch:** 256 tokens/step * 20 accum = 5,120 tokens per weight update.


**Evaluation:** Sliding window eval with stride=64 at seq_len=256.

## Results

| Metric | Value |
|---|---|
| val_loss | 5.4222 |
| val_bpb | **3.2636** |
| Compressed model | 8,830,989 bytes |

## Energy measurements

Power measured using macOS `powermetrics` at 1-second intervals:

| Component | Average power |
|---|---|
| ANE | 1,171 mW |
| CPU | 4,728 mW |
| Combined | 5,958 mW |

## validation

### doesn't require ANE:

```bash
python3 train_gpt.py --eval-only \
    --load-artifact model_artifact.bin \
    --data-dir /.../fineweb10B_sp1024 \
    --tokenizer /.../fineweb_1024_bpe.model
```

### generate artifact from ANE checkpoint

```bash
python3 train_gpt.py --eval-only \
    --ckpt /.../ane_golf_baseline_ckpt.bin \
    --save-artifact model_artifact.bin \
    --data-dir /.../fineweb10B_sp1024 \
    --tokenizer /.../fineweb_1024_bpe.model
```

### Note on training infrastructure

Training uses a compiled Objective-C binary (`./train`) because ANE dispatch requires Apple's private frameworks via `objc_msgSend`. The `train_gpt.py` script can orchestrate the build and training if the ANE repo is accessible, but I didn't test that path end-to-end. The eval and artifact paths work on any platform with PyTorch.

## Thanks to

- [maderix/ANE](https://github.com/maderix/ANE): reverse-engineered ANE APIs and the training pipeline this submission uses, big help throughout this project &
- [maderix substack](https://substack.com/home/post/p-189449078):  his substack!

## Files

| File | Description |
|---|---|
| README.md | This file |
| submission.json | Submission metadata |
| train_gpt.py | Training orchestration + eval bridge + artifact save/load |
| golf_baseline.h | ANE model config (GolfWide: 9L, dim=512, hidden=1024, GQA 8/4) |
| model_artifact.bin | Compressed int8+zlib model weights |
| train_log.txt | Training output from the submitted run |

train_log.txt was captured before the final submission packaging cleanup, so its reported serialized/compressed/code/total byte counts differ slightly from the final submitted files on disk. The evaluated checkpoint and val_bpb=3.2636 are unchanged.
