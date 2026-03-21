# LongContext 4096 + Full SOTA Stack

**Expected val_bpb: ~1.130–1.14** (pending run results)

## Key Insight

The 4096-seq training record (1.2014 BPB) was submitted *before* sliding window eval, FP16 embeddings, 10L+Muon weight decay, Overtone init, or phase-transition resid_mix existed. This submission combines all SOTA techniques with long training context — a combination nobody has tried.

## Techniques

1. **4096 training sequence length**: Model sees 4× more context per token during training. Autoregressive signal is much stronger.

2. **4096 evaluation with stride=256**: Each eval token sees 3840 tokens of context (vs 960 in SOTA's 1024/64). Eval time is the same as SOTA because `64 seqs × 4096 = 256 seqs × 1024` total tokens per batch, and stride scales proportionally.

3. **All SOTA improvements carried forward**:
   - 10 transformer layers (5 encoder + 5 decoder with skip connections)
   - Muon optimizer with decoupled weight decay (WD=0.02)
   - FP16 tied embedding export (avoids int8 errors on both input/output paths)
   - Overtone spectral embedding init (SVD power-law S_k ~ k^{-0.5})
   - Phase-transition resid_mix init (sigmoid-scheduled per layer)
   - Logit softcap = 30
   - NTK-aware RoPE (rope_base=40000 for 4096 training length)

4. **Re-tuned hyperparameters for 4096 context**:
   - Lower LR (matrix=0.025, tied_embed=0.05): longer sequences have larger gradient magnitude
   - Higher Muon momentum (0.98): stabilises fewer but more expensive steps
   - Adjusted warmdown (1600 steps ≈ 24% of ~6700 expected steps)
   - Proportional momentum warmup (400 steps)
   - RoPE base 40000 = 10000 × (4096/1024): same frequency resolution at training length

## Architecture

- Vocab: 1024, Dim: 512, Layers: 10, Heads: 8/4 (GQA), MLP: 2× ReLU²
- Tied embeddings (FP16 export), U-Net skip connections
- ~18.9M parameters, ~14.7 MB artifact (int8+zlib with FP16 embed)

## Run Command

```bash
RUN_ID=longctx4096_fullstack_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_LongContext4096_FullStack/train_gpt.py
```

## Results

*(to be filled after runs)*

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | - | - | - | - |
| 42 | - | - | - | - |
| 7 | - | - | - | - |
| **Mean** | - | - | | |
