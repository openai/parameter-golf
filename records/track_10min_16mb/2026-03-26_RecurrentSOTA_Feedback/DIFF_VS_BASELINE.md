# Differences vs Baseline (LeakyReLU + Legal TTT + Parallel Muon)

Baseline: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
Recurrent: `records/track_10min_16mb/2026-03-26_RecurrentSOTA_Feedback/train_gpt.py`

Everything not listed below is identical (tokenizer, data loading, optimizer, SWA, EMA, TTT algorithm, quantization scheme, attention, MLP, block structure, embeddings).

---

## Architecture

### Layer split changed
- **Baseline**: encoder (first half) / decoder (second half) with skip connections between them
- **Recurrent**: stem (layers 0..core_start-1) / core (layers core_start..core_end-1, looped N times) / tail (layers core_end..num_layers-1) with skip connections between stem and tail

### New: Recurrent core with progressive passes
- Core layers are run `num_passes` times per forward pass
- `PASSES_SCHEDULE` ramps passes during training: `"0:1,4500:2,5500:3,6000:4"`
- `EVAL_PASSES=4` overrides pass count at eval time (train cheap, eval deep)

### New: ResidualScale (inlined from stability.py)
- Learnable per-pass scalar `alpha_k` contracts residual: `h_{k+1} = h_k + alpha_k * delta`
- Init 0.5, learned during training
- Prevents hidden state magnitude growth across passes

### New: ErrorFeedbackModule (inlined from feedback.py)
- Low-rank residual approximation + diagonal correction between passes
- 2560 params (rank=2, dim=512)
- Inactive on pass 0, active on subsequent passes
- **Known bug**: never passed to eval/TTT forward calls, so corrections are absent at inference

### New: Jacobian proxy loss
- Regularization: `lambda * relu(||delta||/||h|| - 1)^2` with lambda=0.1
- Penalizes hidden state growth ratio > 1.0, enforcing contractive dynamics
- Only during training, only on core block

### XSA skips core layers
- Baseline: XSA on last N layers unconditionally
- Recurrent: XSA on last N layers BUT skips core layers (4-6) since they run multiple times

### New: `_fake_quantize` for core weights
- STE-based fake int6 quantization applied to core bank weights during training
- Starts disabled (`CORE_QUANT_ENABLED=0`), auto-enabled when late QAT triggers
- Separate from baseline's `CastedLinear` QAT since core weights come from parameter banks

### New: `_forward_hidden` method
- Shared implementation for both `forward()` and `forward_logits()`
- Returns `(x, h_core_in, h_core_out)` for Jacobian proxy loss
- Both accept `feedback_fn=None, stabilizer=None` kwargs

---

## Training Loop

### Progressive passes schedule
- Each step checks schedule and dynamically updates `base_model.num_passes`
- Pre-compilation: last N warmup steps cycle through each pass count variant to cache `torch.compile` graphs (zero recompilation overhead during training)

### Forward pass signature
- **Baseline**: `loss = model(x, y)`
- **Recurrent**: `loss = model(x, y, feedback_fn=feedback_fn, stabilizer=stabilizer)`

### Late QAT extended
- Adds `step > 100` guard
- Also enables `base_model.core_quant_enabled = True` for bank weight fake quantization

### EMA includes feedback module
- Feedback weights stored with `_fb.` prefix in EMA state
- Separated back out when applying EMA (model gets model keys, feedback gets `_fb.` keys)

### WandB integration (new)
- Conditional logging of train_loss, val_loss, val_bpb, grad_norm, step_avg_ms, lr_scale, growth ratios

### Stability diagnostics
- Per-step recording of h_norms, growth_ratios from stabilizer
- Logged at validation steps, then reset

---

## Post-Training / Evaluation

### Intermediate evaluations removed
- **Baseline**: runs int6 roundtrip eval, sliding window eval (both strides), then TTT
- **Recurrent**: skips all intermediate evals, goes straight to TTT to maximize time budget

### Eval passes override
- After export, `num_passes` changed from training value to `EVAL_PASSES`
- `ResidualScale.scales` padded for extra passes (init 0.5 for new entries)
- `export_sd` re-captured after resize

### No torch.compile on eval model
- **Baseline**: compiles eval model before quantized evaluation
- **Recurrent**: uses eval model directly (no compilation before TTT)

---

## New CLI Arguments

| Argument | Default | Purpose |
|---|---|---|
| `--feedback-mode` | diagonal | identity/diagonal/low_rank/none |
| `--feedback-rank` | 2 | Rank for low-rank components |
| `--per-pass-feedback` | False | Separate correction per pass |
| `--residual-scale-init` | 0.5 | Init value for per-pass scaling |
| `--jacobian-proxy-weight` | 0.01 | Jacobian proxy regularization weight |
| `--no-interpass-rmsnorm` | False | Disable RMSNorm between passes |
| `--clip-hidden` | False | Enable hidden-state clipping |
| `--clip-value` | 10.0 | Clipping threshold |

## New Hyperparameters (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `CORE_START` | 3 | First layer of recurrent core |
| `CORE_END` | 8 | Last layer (exclusive) of recurrent core |
| `NUM_PASSES` | 1 | Initial number of recurrent passes |
| `EVAL_PASSES` | 0 | Override pass count for evaluation (0=use NUM_PASSES) |
| `PASSES_SCHEDULE` | "" | Progressive schedule, e.g. "0:1,4500:2,5500:3,6000:4" |
| `CORE_QUANT_BITS` | 6 | Bit-width for core fake quantization |
| `CORE_QUANT_ENABLED` | 0 | Initial state (auto-enabled by late QAT) |
