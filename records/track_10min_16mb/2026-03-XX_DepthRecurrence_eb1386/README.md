11-layer int6 submission with gated XSA and cosine LN scale for the 10-minute 16MB track.

The foundation is an 11-layer transformer at d=512 with 3x relu-squared MLP, GQA 8/4, and tied embeddings. I started from the observation that the standard XSA approach (hard on/off per layer) leaves information on the table. Instead of binary activation, each layer that uses XSA has a learned sigmoid gate that controls how much of the self-value projection gets subtracted. The gate initializes at 0 (sigmoid=0.5), letting the model discover the right blend during training. This is applied to the last 4 layers.

For the layer norm scaling, I found that the standard inverse-sqrt schedule drops too aggressively for middle layers. I use a cosine decay instead: `cos(pi * i / (2 * L))`, which gives a smoother transition from full scale at layer 0 down to ~0.14 at layer 10. The cosine curve keeps middle layers more active while still dampening the deepest layers where representations tend to saturate.

Bigram context comes from two sources. SmearGate blends each token embedding with the previous token's embedding through a learned per-dimension gate. BigramHash maps consecutive token pairs through an XOR hash into a 4096-bucket embedding table (dim 128, projected up to 512). I went with 4096 buckets instead of the more common 2048 to reduce hash collisions for the 1024-token vocabulary (1024^2 possible bigrams). Both are zero-initialized and scale up gradually.

Quantization uses per-row int6 (range [-32, 31]) for all large weight matrices, fp16 for the tied embedding (quantization errors compound on both input and output sides), and fp32 for control parameters. Compressed with zstd at level 22. Weight decay at 0.04 on both Muon and Adam keeps magnitudes low for clean quantization. Late QAT activates in the final ~4% of training (when lr_scale drops below 0.1) using straight-through estimation, which lets the model adapt to int6 rounding without disrupting early training dynamics. EMA with decay 0.997 replaces SWA for smoother final weights.

Other details: partial RoPE applies rotary embeddings to only 16 of 64 head dimensions, letting the remaining 48 dimensions attend position-free. Orthogonal initialization on all weight matrices with 1/sqrt(2L) scaling on output projections. Muon momentum 0.99 (warmup 0.92 over 1500 steps), matrix LR 0.025, grad clip 0.3, batch 524K tokens, training at seqlen 2048.

Evaluation uses sliding window at stride 64 with seqlen 2048. Each token gets scored with nearly full context. Batched 32 windows at a time across 8 GPUs.

Configuration: `NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 ROPE_DIMS=16 XSA_LAST_N=4 LN_SCALE=1 BIGRAM_VOCAB_SIZE=4096 EMA_DECAY=0.997 LATE_QAT=1 MUON_WEIGHT_DECAY=0.04 MATRIX_LR=0.025 GRAD_CLIP_NORM=0.3`

```bash
RUN_ID=gated_xsa_cosln \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
