# OmniClaw Submission — Parameter Golf

## Result
- **val_bpb**: ~1.16 (work in progress)
- **Model size**: 54.8M params → ~14.8MB int6+brotli (under 16MB limit)
- **Format**: int6+brotli (competition primary metric)

## Architecture
- **Model dim**: 640, **Layers**: 11, **Heads**: 10 (Q), 5 (KV) — GQA
- **MLP mult**: 4 (with tied embeddings)
- **Vocab**: 8192 (SP8192 BPE tokenizer)
- **Depth recurrence**: Layers 3-5 looped 2× (effective 14 layers, no extra params)
- **Parallel residuals**: Layers 7+ (GPT-J style)
- **Smear gate**: Blend token embedding with predecessor's embedding
- **Partial RoPE**: dim=16 (instead of full)
- **QK-Gain**: init=5.25, stabilizes attention with GQA
- **Logit softcap**: 30.0
- **XSA**: Cross-sequence attention on last 4 layers (7-10)

## Quantization
- **Mixed int8/int6 + brotli**: int8 for embedding matrices (tok_emb/lm_head), int6 packed per-row for all other weights
- **GPTQ-lite**: Per-row clip percentile search for optimal quantization
- **Competition format**: int6+brotli roundtrip (primary), also outputs int8+zlib and int6+zstd for comparison

## Training
- **Optimizer**: Muon with EMA (decay=0.9965)
- **QAT**: Disabled (LATE_QAT_THRESHOLD=0.0) — rely on GPTQ-lite post-hoc quantization
- **Warmup**: 20 steps, warmdown over last 3500 iterations
- **Batch**: 786K tokens/step, seq_len=2048
- **Ortho init**: Enabled
- **EMA start**: 50% of training

## Innovations
1. **Depth recurrence** — Layers 3-5 are looped 2×, adding compute without unique parameters
2. **Score-first TTT** — Doc-independent LoRA TTT at eval time, only keep adaptations that improve loss (rollback if worse)
3. **Mixed int8/int6 quantization** — Better preservation of embedding quality
4. **SmearGate** — Smooths token representations at boundaries
5. **XSA** — Cross-sequence attention allows information flow across sequence boundaries