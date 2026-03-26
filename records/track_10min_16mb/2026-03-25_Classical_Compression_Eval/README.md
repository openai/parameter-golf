# Classical Compression Eval-Time Augmentation

**val_bpb: TBD** (pending 8xH100 run) | 8xH100 SXM

## Approach

Novel eval-time augmentation that brings classical data compression techniques (cmix/PAQ) into the neural model evaluation pipeline. All techniques are backward-looking only, zero artifact cost, and run during evaluation.

### Base Model
PR #549 stack: 11L transformer, 512d, XSA on all layers, LeakyReLU(0.5)^2, 3x MLP, Partial RoPE (16/64), EMA + SWA, BigramHash, SmearGate, GPTQ-lite int6 + lzma, Parallel Muon optimizer.

### Novel Eval-Time Techniques

1. **Multi-Order N-gram Backoff (orders 2-7)**: Vectorized numpy implementation using flat uint32 arrays (4M buckets per order). Entropy-adaptive alpha mixing: alpha scales with model uncertainty. Processes segments of tokens in batch, not per-token.

2. **Logistic-Domain Mixing**: Inspired by PAQ/cmix context mixing. Instead of linear interpolation, combines predictions in log-odds space for better handling of extreme probabilities.

3. **Match Model** (planned): cmix-style long-range exact substring matching for repeated boilerplate, templates, and code patterns in web text.

4. **Adaptive Probability Maps (APM)** (planned): PAQ-style error correction lookup tables that learn to fix systematic biases in the mixer output.

## Initial Results (1xH100, 5 iterations, proof of concept)

| Eval Method | val_bpb |
|-------------|---------|
| Standard (int6 roundtrip) | 4.0686 |
| Standard sliding window (stride=256) | 4.0686 |
| **Compressed eval (n-gram + entropy-adaptive)** | **0.5201** |

Note: 5-iteration model is severely undertrained. These numbers demonstrate the compression pipeline works, not final submission quality.

## Credits

- Base model: PR #549 by @abaybektursun (LeakyReLU^2 + Legal TTT + Parallel Muon)
- N-gram eval technique: Inspired by PR #727 by @Asukabot0 (first legal sub-1.0 BPB)
- Classical compression techniques: cmix by Byron Knoll, PAQ by Matt Mahoney
