# Causal Oscillator LM

A language model built from the damped harmonic oscillator equation.

## Architecture

The only learnable transform is H(ω) = 1/(ω₀² - ω² + 2iγω).

Each token drives a bank of 192 oscillators as a physical impulse. The damped impulse response h(t) = e^{-γt} sin(ωd·t) / ωd creates temporal context — recent tokens ring loudly, distant tokens have decayed. Causal convolution via FFT makes this parallel and exact.

Attention layers (12 layers, 16 heads) operate on these physics-enriched oscillator states for long-range dependencies.

## Key differences from transformer baseline

- No positional embeddings — temporal position is encoded by the physics (decay envelope)
- No token embedding lookup — tokens drive oscillators through a learned impulse coupling
- Causal context from physics — the oscillator's ringing IS the context window
- Every parameter is physically interpretable (frequencies in Hz, damping ratios)
- Training is monotonically stable — no loss spikes

## Results

- **Val BPB: 1.338** (best checkpoint at 40K steps on 1xH100)
- **Round-trip BPB: 1.340** (int8 quantization with float16 physics params)
- **Parameters: 14.8M**
- **Compressed size: ~11.2MB**

## Optimizer

Muon for 2D weight matrices, AdamW for oscillator parameters and embeddings. LR=0.003.

## Notes on optimization

This submission was developed on a budget of ~$20 in RunPod credits over 2 days. No hyperparameter sweeps, no architecture search, no multi-seed validation. The model config (192 oscillators, 12 layers, LR=0.003) was found through manual experimentation on a single H100.

The architecture has significant room for improvement: we have not explored larger batch sizes, gradient accumulation tuning, EMA weight averaging, learning rate schedules beyond linear warmup + cosine warmdown, or the many compression tricks (int6, GPTQ, lzma) used by top submissions. The current BPB of 1.34 reflects a novel architecture with minimal optimization, not a tuned system.

## Quantization advantage

The architecture has a structural advantage for compression that we have not yet fully exploited.

The oscillator bank parameters (ω₀, γ) are physics-critical — a small shift in frequency moves the entire resonance curve — but there are only 384 of them (2 per oscillator × 192 oscillators). The attention weights (14.7M parameters) are compression-tolerant. This natural separation means:

- **Current:** float16 physics + int8 attention = 11.2MB, round-trip gap 0.002 BPB
- **Predicted with int4 attention:** float32 physics (~1.5KB) + int4 attention (~7.4MB) = ~7.4MB total
- **At 7.4MB we have 8.6MB of headroom** — enough for a **24M+ parameter model** within 16MB

No other architecture has this clean split between precision-critical and compression-tolerant parameters. Transformer submissions must quantize uniformly or spend engineering effort (GPTQ, mixed-precision sweeps) to find which weights tolerate compression. Our architecture tells you structurally: physics needs precision, coupling doesn't.

With 8xH100 and int4 quantization, we predict:
- 24M params fitting in 16MB (vs current 14.8M)
- 8x batch size enabling ~40K steps in 10 minutes (vs ~5K on 1xH100)
- BPB well below 1.22 baseline, potentially approaching 1.10

We have not validated this on 8xH100 yet — the current result is from 1xH100 with minimal optimization.

## Paper

[Causal Oscillator Network: A Neural Network Built from Damped Harmonic Oscillator Dynamics](https://zenodo.org/records/19326054)

## Code

Architecture source: https://github.com/rolandnsharp/resonance
