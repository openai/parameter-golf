# Record: Parallel Residuals + Mini Depth Recurrence + adaLN

This submission is based on [2026-03-31_ParallelResiduals_MiniDepthRecurrence](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/README.md), with one addition: **adaptive layer norm (adaLN)** conditioned on recurrence iteration.

## Early Results

An initial smoke-test run on **4×H100s for 600s** (50% of the submission compute) reached **val_bpb 1.2551** (val_loss 2.1193 nats) after GPTQ quantization, fitting within the 16 MB size limit (15.26 MB quantized artifact). The run completed only 3,438 steps — recurrence didn't activate until step 3,000, leaving just ~400 steps of recurrent training before the wallclock cap. Despite that, the loss curve was still descending cleanly at cutoff. This is a signs-of-life result: the model trains stably with adaLN enabled and the GPTQ pipeline completes cleanly. A proper 8×H100 full-budget run is needed for a real number.

## adaLN for Recurrence

The core idea is that when a layer is reused across recurrence steps, it sees activations at different "depths" each time — the first pass and the second pass are structurally different contexts. Standard weight tying gives the layer no way to distinguish them. adaLN addresses this with near-zero compute overhead.

**Implementation:**

- A small learned embedding table (`step_emb`) maps each recurrence occurrence index (0, 1, ...) to a `model_dim`-sized vector.
- A single linear projection (`step_adaln`, shape `model_dim → 6 × model_dim`, zero-initialized) produces six per-channel scalars from that embedding.
- At each recurrence pass, the block receives `adaln_params = [γ_attn_in, β_attn_in, γ_attn_out, γ_mlp_in, β_mlp_in, γ_mlp_out]` and applies them as:
  - Pre-attention: `attn_in = attn_in * (1 + γ1) + β1`
  - Post-attention: `attn_out = attn_out * (1 + γ3)`
  - Pre-MLP: `mlp_in = mlp_in * (1 + γ2) + β2`
  - Post-MLP: `mlp_out = mlp_out * (1 + γ4)`
- Non-recurrent layers receive `adaln_params = None` and are unaffected — zero overhead on the majority of the forward pass.
- The projection weight is zero-initialized, so at init all modulations are identity. Training starts from the same point as the baseline.

**Parameter cost:** `model_dim + 6 × model_dim × num_occurrences`. With `model_dim=512` and two recurrence occurrences, this is `512 + 6 × 512 × 2 = 6,656` parameters — negligible relative to the 16 MB budget.

**Why it helps:** The recurrent layers (4, 5) are applied twice. Without adaLN, both passes use identical pre/post-norm scaling, forcing the layer to be a jack-of-two-depths. With adaLN, each pass gets its own lightweight affine shift on top of RMSNorm, giving the shared weights the information they need to specialize without doubling their count.

## Reproducibility

```bash
SEED=$SEED POST_GPTQ_EVAL_ONLY=0 BIGRAM_DIM=112 MIXED_QUANT=1 N_INT6_LAYERS=32 \
NUM_LAYERS=11 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 REPEAT_UNTIE_MLP=full \
REPEAT_UNTIE_MLP_LAYERS=4,5 DISABLE_LAYER0_ATTN=1 PARALLEL_RESIDUAL=1 \
PARALLEL_START_LAYER=7 FILM_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`brotli` must be installed for the final artifact path.
