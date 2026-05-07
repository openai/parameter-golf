# Approach: Blackice architectural enhancements for parameter-golf

We apply techniques developed during our Blackice Diffusion Language Model research to the parameter-golf GPT baseline, focusing on zero-parameter compute enhancements and training improvements.

## Architecture (from Blackice DLM research)

- **Block Attention Residuals** (Moonshot AI, arXiv 2603.15031) — Replaces the baseline's fixed skip_weights with dynamic softmax attention over encoder layer outputs, enabling learned depth-wise information routing
- **Per-head gated attention** — Learnable sigmoid gate per attention head, preventing attention-sink pathology at near-zero parameter cost
- **Looped middle blocks** (NanoGPT Slowrun) — Repeats layers 4-7 twice per forward pass, adding computational depth without parameters (critical for the 16MB size cap)
- **11 layers with 3x MLP expansion** — Increased capacity matching top leaderboard configurations

## Training improvements

- **EMA** (decay=0.995) — Exponential moving average weights for final evaluation
- **Cosine LR decay** — Smoother schedule than baseline's linear warmdown, beneficial for wallclock-capped training
- **Quantization-Aware Training** (last 15%) — Simulates int8 rounding during training to minimize the float-to-quantized BPB gap

## Local validation

- Device: M4 Max, 500 steps
- val_bpb: 1.475 (float), int8 roundtrip: 1.648
- Artifact size: 13.7MB (2.3MB under 16MB cap)
- Loss still dropping fast at step 500 — full 20K step H100 run needed for competitive BPB
