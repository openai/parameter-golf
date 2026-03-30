# Record: 1.1194 BPB — v9 Batched Muon + Full GPTQ + Random Calibration

**11L Batched Newton-Schulz Muon + XSA-all + Full GPTQ (Random Calib) + FA3 + LZMA + Stride-64 Sliding Eval**

**val_bpb: 1.1194** (3-seed mean sliding) | **15.90 MB** max artifact | 8xH100 SXM, 600s

![v9](v9.png)

## Results (3 seeds, 8xH100 SXM, Nebraska)

| Seed | Sliding BPB | Post-EMA BPB | Steps | ms/step | Artifact |
|------|-------------|-------------|-------|---------|----------|
| 1337 | 1.1191 | 1.1368 | 6,893 | 87.05 | 15,899,949 bytes |
| 42 | 1.1195 | 1.1374 | 6,898 | 86.99 | 15,976,341 bytes |
| 7 | 1.1195 | 1.1373 | 6,902 | 86.94 | 15,898,969 bytes |
| **Mean** | **1.1194** | **1.1372** | **6,898** | **87.0** | - |
| **Std** | **0.0002** | **0.0003** | - | - | - |

## Key Innovation: Batched Newton-Schulz Orthogonalization

The primary technical contribution is **batched Muon optimizer acceleration** via `torch.bmm`. Instead of processing 66 weight matrices through Newton-Schulz iterations sequentially, we group them by shape into 4 batches and orthogonalize in parallel:

```
qo_group:       22 matrices of (512, 512)    -> one bmm call
kv_group:       22 matrices of (256, 512)    -> one bmm call
mlp_up_group:   11 matrices of (1536, 512)   -> one bmm call
mlp_down_group: 11 matrices of (512, 1536)   -> one bmm call
```

**Result:** 5% optimizer speedup on 1xH200 (604ms vs 636ms/step), translating to ~400 additional training steps over 600s on 8xH100 SXM.

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 11 (U-Net: 5 encoder + 6 decoder) |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP | 3x width, LeakyReLU(0.5)^2 |
| Params | 26,993,756 |
| XSA | All 11 layers |
| Embeddings | BigramHash(2048, 128) + VE128(layers 9,10) |
| Attention | FlashAttention-3 (Hopper native) |
| Position | Partial RoPE (16/64 dims) |
| Other | SmearGate, LN Scale, Logit Softcap(30) |

## Quantization: Full GPTQ with Random Calibration

We use Full Hessian GPTQ (Frantar et al.) with a key compliance innovation: **random token calibration**. Instead of reading training data for Hessian collection (which raises compliance questions about post-training data access), we generate random tokens from the vocabulary distribution:

```python
class RandomCalibLoader:
    def next_batch(self, batch_tokens, seq_len, grad_accum_steps):
        tokens = torch.randint(0, vocab_size, (n_seqs, seq_len + 1), device=device)
        return tokens[:, :-1], tokens[:, 1:]
```

This produces representative activations for Hessian estimation without accessing training data during the export phase. The quality loss vs training-data calibration is negligible (~0.0003 BPB per PR #1019's findings).

- Int6 per-row quantization with Hessian error compensation
- Column reordering by Hessian diagonal (actorder)
- Block-wise Cholesky compensation (block_size=128)
- 5-percentile clip search for per-row scales
- LZMA compression (preset=9, extreme)

## Research Journey

This submission is the result of extensive research and ablation testing across multiple AI systems (Claude, ChatGPT, Gemini Pro). We explored 9 research directions before arriving at v9:

![Research Journey](beam.png)

### What We Explored (and Why We Moved On)

**JEPA (Joint-Embedding Predictive Architecture)** — 14 ablation tests across two H200 sessions:
- Vanilla JEPA with separate context/target/predictor encoders: -0.019 to -0.058 BPB penalty
- Discovered critical gradient interference bug (JEPA gradients fighting backbone via un-detached hidden states)
- After detach fix: penalty reduced 67% (0.058 -> 0.019) but still net negative
- Tested 6 JEPA weight configurations, 3 span configurations, 2 EMA decay values
- Conclusion: At 600s/7000 steps, the compute overhead outweighs regularization benefit for 27M-param models

**STP (Semantic Tube Prediction)** — LeCun lab's Feb 2026 variant (arXiv 2602.22617):
- Zero parameters, negligible compute, gradient flows into backbone
- Still hurt at 90s training (-0.075 to -0.117 BPB depending on weight)
- Regularization benefits can't overcome the gradient interference at short training budgets

**Label Smoothing** — Found and fixed an eval contamination bug (smoothing was applied during eval via model.forward):
- After fix: still hurts at short training (-0.090 BPB at 90s)

**Legal Score-First TTT** — Implemented but disabled:
- PR #1019 demonstrated TTT is ineffective on XSA-all stacks (25 failed attempts)
- XSA-all already captures the inter-document patterns TTT would adapt to

### What Worked

1. **Batched Muon** (this submission) — 5% faster optimizer, ~400 extra steps
2. **Full GPTQ with random calibration** — better quantization than GPTQ-lite, fully compliant
3. **LZMA compression** — ~5% smaller artifact than zstd, keeping us under 16MB
4. **JEPA OFF, STP OFF, TTT OFF, LS OFF** — removing all auxiliary losses let the model focus 100% on next-token prediction

## Compliance Checklist

- [x] 3 seeds on 8xH100 SXM (Nebraska, Vast.ai)
- [x] All seeds train in <= 600s
- [x] All artifacts <= 16,000,000 bytes
- [x] No test-time training on validation data
- [x] No training data access during quantization (random calibration tokens)
- [x] No network calls during evaluation
- [x] No external compute
- [x] Single file (train_gpt.py, 2,014 lines)

## Run Command

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
GPTQ_ENABLED=1 STP_ENABLED=0 TTT_ENABLED=0 LABEL_SMOOTHING=0.0 \
XSA_LAST_N=11 EVAL_STRIDE=64 SEED=1337 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Acknowledgments

Built on the excellent foundation of the parameter-golf community. Techniques borrowed from PRs #414 (signalrush), #549 (abaybektursun), #399 (Parameter Banking concept), #1019 (random GPTQ calibration insight). Multi-AI research collaboration: Claude Opus (implementation + testing), ChatGPT (compliance review), Gemini Pro (deep research on JEPA/STP theory).
