# AR self-generated GPTQ calibration

**Status:** candidate — post-training, hotstart-screenable
**Expected Δ:** +0.001 to +0.005 bpb (hopeful; claimed −0.0078 nats ≈ −0.0046 bpb in the source submission, but bundled with other changes)
**Source:** `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md`. The calibration-data change is one piece of a multi-change submission; isolated Δ not cleanly reported.

## Idea
Current GPTQ calibration uses **FineWeb training data** (`ShuffledSequenceLoader`, 64 batches) to compute per-matrix Hessians (`train_gpt_sota.py:963-964`, `collect_hessians` L763-806). The 2026-03-25 submission replaced this with **model-autoregressively-generated text** as calibration data: sample ~64 sequences × 2048 tokens from the trained model itself at temperature 0.8, then feed those through the Hessian-collection pass.

The Hessian then reflects the actual token distribution the model will encounter at inference time (the model's own output distribution), rather than the training-data distribution (which differs from the model's predictions, especially on tail tokens).

## Why it might help
- **Calibration-distribution mismatch is a known GPTQ pain point.** Per-matrix Hessian `X^T X` is computed from input activations X; if X at inference differs from X at calibration, the GPTQ error correction (Cholesky-based) is misfitted.
- SOTA's `collect_hessians` draws X from the training distribution, but the submission is scored on val data via sliding-window eval. Shifting calibration toward the model's own output distribution (which matches the autoregressive eval pattern) may give tighter quantization.
- **Legal and self-contained.** No external data required. The model generates its own calibration text.
- **Not a per-row scale tweak** — this is the first post-training candidate that isn't in the class that failed twice (Hessian-SDClip, SWA+EMA both hurt). Different mechanism.

## Code-change sketch
In spec-001-style sweep wrapper:
1. Load checkpoint, apply EMA (standard hotstart flow).
2. **Generate calibration text:** sample 64 sequences × 2048 tokens from the model at `temperature=0.8`. Concatenate into a `torch.Tensor` of shape [batches, seq_len].
3. Build a custom loader that yields these generated sequences (not `ShuffledSequenceLoader` which yields FineWeb).
4. Call existing `collect_hessians(base_model, custom_loader, h, device, n_calibration_batches=64)`.
5. Run standard `gptq_mixed_quantize` + Brotli + eval.

Generation function already exists or is simple: a greedy/sampling loop over the model's forward pass. Probably 30-50 lines of new code total.

Env toggle: `GPTQ_CALIB_MODE=selfgen` vs `GPTQ_CALIB_MODE=fineweb` (default).

## Hotstart screening plan
**Post-training change, no training needed.**

- **Hotstart from:** `ckpt_final_pre_ema_step3849.pt` (spec 000).
- **Pipeline:** load ckpt → apply EMA → generate 64×2048 tokens → collect Hessians from generated text → GPTQ → Brotli → quantized eval.
- **Cost per run:** ~5-8 min on 1×H100 (add ~3-5 min for AR generation on top of standard GPTQ+eval).
- **Control:** our existing spec-001 λ=0 number (1.10518) is the fineweb-calibrated baseline. Same checkpoint, so apples-to-apples.
- **Screen cost:** ~$1 on 1×H100.

## Configurations worth testing
1. **C0:** fineweb calibration (control, reproduces 1.10518).
2. **C1:** AR self-gen, temp=0.8, 64 × 2048 tokens (direct port from submission).
3. **C2:** AR self-gen, temp=1.0 (more diversity).
4. **C3:** AR self-gen, temp=0.5 (more deterministic, closer to model's high-confidence predictions).
5. **C4:** 50/50 mix of fineweb + AR self-gen (hedge against the AR distribution being too narrow).

5 configs × ~6 min = ~30 min wall, ~$1-2 total.

## Risks / open questions
- **Generation-quality dependency.** If the model's AR output is garbage at some random seed, calibration is garbage. Need to verify text looks reasonable before trusting the Hessian.
- **Feedback loop.** The model is calibrating on its own outputs, which may amplify whatever biases it already has. Could be worse than fineweb calibration if those biases hurt quantization.
- **Temperature sensitivity.** Wrong temp could produce too-narrow or too-broad activation distributions. The sweep above covers this.
- **AR generation time.** 64 seqs × 2048 tokens is 128K tokens. On 1×H100 with compile cache warm, probably ~2-5 min. Tolerable.
- **Did near-SOTA's −0.0046 bpb come from this change specifically, or from the bundled BigramHash-3072 / XSA?** The 03-25 README doesn't isolate. We're guessing that calibration-data contributes meaningfully.
- **Memory.** Generation pass may need KV-cache; ensure we don't OOM on 1×H100 with the 70M model.

## If this works
- First post-training candidate that doesn't hit the per-row-scale-tweak dead end.
- Stacks with every training-time candidate (spec 003 BigramHash, future SwiGLU, etc.) — it's a quant-pipeline change.
- If the Δ is real (even +0.001), it's the cheapest "guaranteed" improvement we can get — no new training, just re-quantize.

## Priority
**High.** This is my most-interesting remaining candidate after BigramHash. If spec 003 shows signal, this is a natural follow-up/stacking check. If spec 003 doesn't, this might be the last cheap shot at a post-training win before we commit to full-retrain experiments (layerwise LR, SwiGLU).
