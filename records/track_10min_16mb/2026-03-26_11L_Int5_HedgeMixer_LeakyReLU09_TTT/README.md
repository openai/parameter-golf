record submission on 8xH100 SXM 80GB. 10-minute training + 10-minute eval.

val_bpb: 1.1105 (post int5/int6 + zstd + TTT + HedgeMixer)
pre_quant_val_bpb: 1.1342

## architecture

11 layers, 512 dim, 8 heads, 8 KV heads, 3.5x MLP (hidden=1792), LeakyReLU(0.9)^2, tied embeddings, logit softcap=30.

GatedAttention (per-head learned scalar gate), ValueResidual (per-block learned x0 injection), XSA on all 11 layers, SmearGate, BigramHash(8192, dim=128), ValueEmbedding (dim=128) on layers 9-10, Partial RoPE (16/64), LN Scale, OrthoInit + muP, U-Net encoder-decoder with learned skip weights.

## training

Muon WD=0.04, AdamW WD=0.04, matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035, momentum 0.92->0.99 warmup 1500 steps, warmdown 3500 iters, batch 786432 tokens, seq_len 2048, grad_clip 0.3, EMA(0.997) + Tight SWA (scale<0.2, every 50), Late QAT soft-round STE int5 at scale<0.5, CROWN-Q regularization. 5684 steps at 101ms/step, 582s training.

## quantization

GPTQ Hessian-based (128 calibration samples), int5 all blocks, 3% magnitude pruning, zstd-22. FP16 passthrough for tied embedding. artifact: 15,951,599 bytes.

## eval

sliding window stride=64, seq_len=2048. score-first TTT: 4 epochs AdamW lr=0.0005, freeze first 2 blocks, byte-weighted loss, Polyak averaging (0.998), adaptive cosine LR.

6-expert Hedge context mixer: neural, unigram, bigram, trigram (65536 hash), 4-gram (32768 hash), neural entropy. multiplicative weights with adaptive eta. n-gram tables built online from scored tokens. eval time: 570s.

## results

| metric | value |
|--------|-------|
| val_bpb (TTT + HedgeMixer) | 1.1105 |
| pre-quant val_bpb | 1.1342 |
| steps | 5684 |
| ms/step | 101.4 |
| artifact | 15.95 MB |
| train time | 582s |
| eval time | 570s |

## command

```bash
pip install zstandard flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

RUN_ID=v3_8gpu SEED=1337 MAX_WALLCLOCK_SECONDS=600 TTT_EPOCHS=4 \
TTT_LR=0.0005 TTT_FREEZE_BLOCKS=2 TTT_CHUNK_TOKENS=32768 \
TTT_OPTIMIZER=adamw USE_MIXER=1 MIXER_ETA=0.1 USE_POLYAK=1 \
BYTE_WEIGHTED_TTT=1 ADAPTIVE_LR=1 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

single seed (1337). additional seeds pending compute.
