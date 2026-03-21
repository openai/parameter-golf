# Technique Origins & Adoption

| Technique | First PR | Originator | Adoption |
|-----------|----------|------------|----------|
| Sliding Window Eval | #50 | @mattqlf | Near-universal (20+) |
| FP16 Tied Embedding | #42 | @chonchiog | ~10+ |
| Int6 Quantization | #39 | @nanlliu | ~15+ |
| MLP 3x Expansion | #70 | @jfprincz | ~12+ |
| Muon Weight Decay | #60 | @notapplica (from modded-nanogpt) | Several |
| Overtone Spectral Init | #60 | @notapplica | @peytontolbert, @TevBenji (superseded) |
| SmearGate / BigramHash | #102 | @unnir | 17+ adopters |
| OrthoInit | #135 | @unnir | Near-universal among top SmearGate entries |
| Test-Time Training | #77 | @samacqua (LoRA) | 10+ adopters (SGD, causal, Reptile variants) |
| NorMuon | Multiple | Convergent | @mtybadger, @vmfunc, @dexhunter, others |
| QAT with STE | Multiple | Convergent | @rsavitt, @yahya010, @trovatochris, others |
| SWA | #89 | @vmfunc | @mtybadger, @dexhunter, others |
| Int5 MLP Quantization | #76 | @unixmadtoonslab | @thwu1 (SOTA #180) |
| BigramHash Scaling (10240+) | #180 | @thwu1 | @andrewgcodes (16384) |
| Low-Rank Q Factorization | #215 | @JayCheng113 | Novel, no adopters yet |
| Partial XSA | #265 | @unnir | 6+ adopters including @jfprincz (#287, #315) |
| EMA Weight Averaging | #95 | @MatoTeziTanka | @jfprincz, @dennisimoo, @saml212 |
| Reptile Meta-TTT | #296 | @sseanliu | @JackYoung27 (#302) |
| Partial RoPE | #315 | @jfprincz | @Ananddna (#327), @saml212 (#332) |
| LN Scale (1/sqrt(layer)) | #315 | @jfprincz | @saml212 (#332) |
| Late QAT (last 4%) | #315 | @jfprincz | Novel, no adopters yet |
| Gradient-Guided Quant | #332 | @saml212 | Novel |
| TrigramHash | #327 | @Ananddna | Novel |
| Per-Head Temperature | #327 | @Ananddna | Novel |
| BitNet b1.58 | #126, #139 | @Athenox14, @ksang123 | Two independent |

## Key Diffusion Patterns

- **Core five** spread within first 24 hours and became table stakes
- **SmearGate + OrthoInit** became the dominant architecture add-on (~17 adopters)
- **XSA** rapidly adopted after @unnir's #265 (6+ adopters in ~1 day)
- **Reptile meta-TTT** has only 1 adopter so far — high-EV but complex
- **#315's trio** (Partial RoPE, LN Scale, Late QAT) beginning to spread to @saml212's 12L attempt
