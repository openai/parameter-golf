# First Text Diffusion Submission: 1.4584 BPB — Masked Diffusion Language Model (MDLM)

Causal MDLM Training + SUBS Parameterization + ELBO-Weighted Loss + Stratified t Sampling + Zero Mask Probabilities + 11L Transformer + int6 GPTQ + sliding window eval

**val_bpb: 1.4584 (seed=42)** | 11.55 MB artifact | 8×H100 SXM, 555s training + 213s eval

## Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| Sliding BPB | 1.4584 |
| val_bpb (pre-quant) | 1.4855 |
| val_loss | 2.4625 |
| Steps | 2,212 |
| ms/step | 250.91 |
| Training time | 555s |
| GPTQ time | 45s |
| Eval time | 213s |
| Peak memory | 33,778 MiB |
| Artifact | 11,547,850 bytes (11.55 MB) |
| Model bytes | 11,474,336 |
| Code bytes | 73,514 |
| Parameters | 26,994,428 (vocab expanded by 1 for [MASK]) |

## Method

Implements the continuous-time NELBO from Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024), adapted for causal language modeling.

### Training (MDLM objective — 50% of steps)

For each diffusion training step:
1. Sample masking rate `t ~ U(0,1)` with **stratified sampling** (low-discrepancy, reduces ELBO variance)
2. Compute `α_t = 1 - t` (linear noise schedule)
3. Mask each input token independently with probability `(1 - α_t)`
4. Forward pass with masked input (causal attention)
5. Compute cross-entropy loss **only on masked positions**
6. Weight by ELBO coefficient: `1/(1-α_t) = 1/t` (from continuous-time NELBO, Eq. 14)

### MDLM Properties Implemented

| Property | Implementation |
|----------|---------------|
| SUBS: carry-over unmasking | Loss only on masked positions (unmasked = free) |
| SUBS: zero masking probabilities | `logits[:, :, mask_id] = -inf` |
| Continuous-time NELBO | ELBO weight `1/t` for linear schedule |
| Stratified t sampling | `t = (t_base + i/B) % 1` for lower variance |
| Mixed training | 50% diffusion steps, 50% standard NTP |

### Eval (standard autoregressive)

At eval time, no masking is applied. Standard next-token prediction with sliding window. The diffusion training acts as a regularizer that improves representations by forcing robustness to partial information.

### Adaptation for Causal LM

The original MDLM uses bidirectional (encoder-only) attention. We adapt it for causal attention to maintain compatibility with autoregressive evaluation. This is a deliberate design choice — the causal model can still be evaluated autoregressively while benefiting from the diffusion training signal during training.

## Architecture

- Standard 11L causal Transformer
- Vocab expanded by 1 for [MASK] token (1025 total for SP1024)
- diffusion_mix = 0.5 (alternating diffusion and NTP steps)
- All other features from v50: LeakyReLU(0.5)² MLP, BigramHash, SmearGate, GPTQ, etc.

## Command

```bash
TORCH_COMPILE_DISABLE=1 \
DIFFUSION_ENABLED=1 \
DIFFUSION_MIX=0.5 \
NGRAM_EVAL=0 \
KNN_LAMBDA=0 \
SEED=42 \
python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact ≤16,000,000 bytes (11,547,850)
- [x] Training ≤600s on 8×H100 SXM (555s)
- [x] Eval ≤600s (213s)
- [x] GPTQ calibration inside training budget (45s, on training data)
- [x] No validation data during training
- [x] No network calls during evaluation
- [x] No external compute
- [x] No n-gram cache or kNN (clean sliding window eval only)
- [x] Reproducible from `train_gpt.py`

## References

- MDLM: [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) (Sahoo et al., NeurIPS 2024)
- Scaling MDLMs: [arXiv:2410.18514](https://arxiv.org/abs/2410.18514)

## Included Files

- `train_gpt.py` — full training script
- `train_seed42.txt` — training log
- `submission.json` — metadata
- `run.sh` — reproduction script
- `requirements.txt` — dependencies
