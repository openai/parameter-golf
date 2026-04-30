# BitNet-1.58 Ternary HydraMLP

**Track:** non-record, 10-minute wallclock, 16 MB artifact cap
**Tokenizer:** SentencePiece BPE, vocab=1024 (`fineweb_1024_bpe.model`)
**GPU:** 1×H100
**Final exact int4+ternary+zlib roundtrip BPB:** **1.36406657**
**Self-contained package size:** **12.923 MB** (model payload plus included source helpers)

---

## What this submission demonstrates

Replacing int4 fake-quantization on the HydraMLP `gate_up` and `down` weights with **BitNet-1.58 ternary {-1, 0, +1} fake-quantization**. Ternary weights free several MB of artifact budget versus the comparable int4 local-global baseline at the cost of roughly `+0.029` BPB.

Use case: when the 16 MB submission cap is the binding constraint and you want to spend the freed budget on architecture capacity rather than weight precision.

### Quantitative Result

| Variant | val_bpb (1×H100, 600 s) | artifact size |
|---|---|---|
| Comparable int4 local-global baseline | 1.3419 | 17.72 MB, over cap |
| **Ternary HydraMLP package** | **1.36407** | **12.92 MB, under cap** |

Net trade: **about +0.029 BPB for about 4.8 MB of artifact headroom**.

## Mechanism

`TernaryLinear` replaces `QATLinear` for the two HydraMLP projections. Two scale modes are implemented; `buffered_absmean` is the default and is used for this run:

- **`absmax`** — recompute `scale = absmax(group)` every forward pass. Idempotent under roundtrip, sparser (~21% nonzero on Gaussian weights).
- **`buffered_absmean`** — `scale = absmean(group)` during training, persisted to a `ternary_scale_buf` so eval reads the same scale used at training. Density ~69% (BitNet b1.58), exact roundtrip.

Forward (STE backward):
```python
scale = w.abs().reshape(-1, group_size).mean(dim=-1, keepdim=True).clamp(min=1e-10)
w_q = (w / scale).round().clamp(-1, 1)
w_dq = (w_q * scale).reshape(orig_shape)
quant_error = (w_dq - w).detach()
return w + noise_scale * quant_error
```

Export packs each ternary group as `(int8_signs, fp16_per_group_scale)`. The `ternary_scale_buf` is round-tripped exactly because it is stored in the passthrough payload and re-bound to the layer at restore.

## Architecture

Bifurcated local/global recurrent LM:
- `shared_dim=768`, tied embeddings
- Local: `LOCAL_DIM=512`, `LOCAL_LAYERS=8`, `LOCAL_HEADS=4`, sliding-window attn (`window=128`), `LOCAL_MLP_MULT=3.25`
- Global: `GLOBAL_DIM=512`, 1 effective layer, `GLOBAL_HEADS=8`, `GLOBAL_KV_HEADS=4`, GQA, pool ratio 8
- SAGE-Global injection at after-block 3, no SAGE-Bus
- PMI channel encoder/decoder rank 128, basis frozen, aux weight 0.002
- Pointer-generator copy head (logmix v2)
- Bigram + trigram CP priors loaded as additive logit biases (also int4-roundtripped)

Optimizer:
- Polar Express 5-step Newton-Schulz Muon (with safety-factor 1.05) for hidden matrix params
- AdamW for embeddings + heads + scalars + auxiliary tensors
- `MUON_MIN_PARAMS=32768`, `same_shape_batch=True`, replicated bucket size 16 MB

## Reproducing

### 1. Build the bigram + trigram priors (≈5–10 min on CPU/1 GPU)

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
VOCAB_SIZE=1024 \
PMI_RANK=32 \
python priors.py
```

This writes `bigram_logprior.pt`, `bigram_lowrank.pt`, and `trigram_cp.pt` into the working directory.

### 2. Train (1×H100, 600 s wallclock cap)

```bash
GRAD_ACCUM_STEPS=8 \
LOCAL_BATCH_TOKENS=86016 \
TRAIN_BATCH_TOKENS=688128 \
TERNARY_ENABLED=1 \
TERNARY_SCALE_MODE=buffered_absmean \
RUN_ID=bitnet_ternary_hydramlp_1g \
torchrun --standalone --nproc_per_node=1 train_gpt.py \
  2>&1 | tee logs/bitnet_ternary_hydramlp_1g.txt
```

The ternary defaults match this submission. To reproduce the int4-only baseline for comparison, set `TERNARY_ENABLED=0`.

## Files in this submission

- `train_gpt.py` — model + training script (self-contained)
- `priors.py` — bigram + bigram-lowrank + trigram CP builder (single entry point)
- `logs/bitnet_ternary_hydramlp_1g.txt` — the actual training run log used for the reported numbers
- `submission.json` — track metadata + BPB
- `results.json` — full quant byte breakdown + schedule

## Citations

- Wang, S., et al. *BitNet b1.58: Training Ternary {-1, 0, +1} Large Language Models.* 2024.
- Levy, O., Goldberg, Y. *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS 2014. (For SPPMI / SVD background; see `priors.py`.)
- Jacob, B., et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. (For the QAT framework `TernaryLinear` extends.)
