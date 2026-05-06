# Co-occurrence QK Initialization

## Score: val_bpb = 1.3525 (1×H100, single seed)

Trained on 1×H100 80GB in 600 seconds. 15.55MB artifact (int6+zstd). Run on 1×H100 due to compute constraints. Built upon [PR #623](https://github.com/openai/parameter-golf/pull/623).

## Approach

Initializes W_Q and W_K in layer 0 from bigram co-occurrence statistics so that the initial attention pattern reflects real token relationships rather than random noise.

### Mathematical formulation

**Goal**: At step 0, hidden states h ≈ E[token_id] (just the embedding). We want the attention logit between tokens i and j to approximate their co-occurrence:

```
q_i · k_j = (h_i W_Q^T) · (W_K h_j) ≈ C[tok_i, tok_j]
```

where C is the 1024×1024 bigram co-occurrence matrix.

**Step 1 — Build co-occurrence matrix**: Scan 2M training tokens. For each consecutive pair (t_i, t_{i+1}), increment C[t_i, t_{i+1}]. Apply log-transform: C ← log(C + 1) then center (subtract row/column means). This gives a PMI-like matrix.

**Step 2 — Project into model dimension**: Since W_Q and W_K operate on model_dim (512), not vocab_size (1024), we project C into model space via a fixed random matrix P ∈ R^{1024×512}:

```
C_proj = P^T C P    ∈ R^{512×512}
```

**Step 3 — SVD factorization**: Decompose C_proj = U S V^T. Take top d_head (64) components:

```
W_Q ← (U[:, :q_dim] · diag(√S[:q_dim]))^T    ∈ R^{q_dim × 512}
W_K ← (V[:k_dim, :] · diag(√S[:k_dim]))^T     ∈ R^{k_dim × 512}
```

This ensures W_Q^T W_K ≈ C_proj (scaled), so Q·K^T at step 0 reflects co-occurrence.

**Step 4 — Scale normalization**: Rescale W_Q and W_K to match the norm of the default orthogonal initialization, preventing gradient scale mismatch:

```
W_Q ← W_Q · (‖W_Q_orig‖ / ‖W_Q‖)
W_K ← W_K · (‖W_K_orig‖ / ‖W_K‖)
```

**Head diversity**: With 8 heads (head_dim=64), SVD components 1–64 go to head 0, 65–128 to head 1, etc. Each head captures a different slice of co-occurrence structure.

Zero extra parameters — only changes initialization. Co-occurrence computation takes <3s.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 11 (10 unique) |
| model_dim | 512 |
| mlp_activation | ReLU² |
| cooc_init_tokens | 2,000,000 |
| cooc_init_layer | 0 only |
| train_batch_tokens | 524,288 |
| matrix_lr / scalar_lr | 0.025 |
| swa_every | 50, start_frac=0.2 |

## Key Metrics

- **val_bpb: 1.3525** (post int6+zstd roundtrip)
- Pre-quant val_bpb: 1.3245
- Quantization penalty: 0.0280 bpb
- Training: 1,099 steps in 600s (546 ms/step)
- Artifact size: 15,545,987 bytes (15.55MB)
- SWA: averaged 12 checkpoints
- Peak memory: 14,656 MiB
