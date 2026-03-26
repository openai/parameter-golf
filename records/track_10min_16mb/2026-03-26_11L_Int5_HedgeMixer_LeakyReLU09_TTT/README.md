record submission on 8xH100 SXM 80GB. 10-minute training + 10-minute eval.

val_bpb: 1.1105 (post int5/int6 + zstd + TTT + HedgeMixer)
pre_quant_val_bpb: 1.1342

## architecture

11 layers, 512 dim, 8 heads, 8 KV heads, 3.5x MLP (hidden=1792), LeakyReLU(0.9)^2, tied embeddings, logit softcap=30.

- GatedAttention: per-head learned scalar gate
- ValueResidual: per-block learned injection of initial embedding
- XSA (exclusive self-attention) on all 11 layers
- SmearGate + BigramHash(8192, dim=128)
- ValueEmbedding (dim=128) on layers 9-10
- Partial RoPE (16 of 64 head dims)
- LN Scale (1/sqrt(layer+1))
- OrthoInit + muP output scaling
- U-Net encoder-decoder with learned skip weights

## training

- Muon WD=0.04, AdamW WD=0.04
- matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035
- momentum 0.92 -> 0.99 warmup over 1500 steps
- warmdown 3500 iters (wallclock-based)
- batch 786432 tokens, seq_len 2048, grad_clip 0.3
- EMA (decay=0.997) + Tight SWA (scale<0.2, every 50 steps)
- Late QAT: soft-round STE int5, enabled at scale<0.5
- CROWN-Q regularization during warmdown
- 5684 steps at 101ms/step, 582s training

## quantization

- GPTQ (Hessian-based, 128 calibration samples) for all linear layers
- int5 for all blocks (clip_range=15), mixed with int6 option
- 3% magnitude pruning before quantization
- zstd level 22 compression
- FP16 passthrough for tied embedding and control tensors
- artifact: 15,951,599 bytes (15.95 MB)

## eval

sliding window stride=64, seq_len=2048.

score-first TTT: score each chunk under inference_mode, then train on scored tokens. 4 epochs AdamW (lr=0.0005), freeze first 2 blocks, unfreeze norms/scales/head. byte-weighted loss. Polyak averaging (decay=0.998). cosine LR with adaptive scaling.

6-expert Hedge context mixer:
- expert 0: neural model (log-softmax)
- expert 1: online unigram counts
- expert 2: online bigram counts
- expert 3: online trigram hash (65536 buckets)
- expert 4: online 4-gram hash (32768 buckets)
- expert 5: neural entropy (confidence signal)

multiplicative weights update with adaptive eta (decays as more tokens scored). cold start: pure neural until 10K tokens. n-gram tables are GPU-resident, built incrementally from already-scored validation tokens. no information leakage.

adaptive temperature: confident predictions sharpened more (base temp=0.98).

eval time: 570s (within 10-minute budget).

## results

| metric | value |
|--------|-------|
| pre-quant val_bpb | 1.1342 |
| final val_bpb (TTT + HedgeMixer) | 1.1105 |
| training steps | 5684 |
| ms/step | 101.4 |
| artifact size | 15.95 MB |
| peak memory | 26.2 GB per GPU |
| train time | 582s |
| eval time | 570s |

## command

```bash
pip install zstandard flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

RUN_ID=v3_8gpu_submission SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
TTT_EPOCHS=4 TTT_LR=0.0005 TTT_FREEZE_BLOCKS=2 TTT_CHUNK_TOKENS=32768 \
TTT_OPTIMIZER=adamw USE_MIXER=1 MIXER_ETA=0.1 USE_POLYAK=1 POLYAK_DECAY=0.998 \
BYTE_WEIGHTED_TTT=1 ADAPTIVE_LR=1 ADAPTIVE_LR_MAX=3.0 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## included files

- train_gpt.py: training script
- train.log: training + eval log
- submission.json: metadata
