Depth recurrence submission for the 10-minute 16MB track.

The central observation is that the baseline is wildly over-trained at 420 tokens per parameter, and the 10-minute wallclock means you hit diminishing returns on training steps fast. So instead of spending parameters on 11 unique layers that each get used once, I use 8 unique blocks and loop them twice, getting 16 effective layers of processing from 8 blocks worth of stored weights. The model refines its representations iteratively through the same learned transformations, with a per-block residual mix gate maintaining a highway back to the original embeddings for gradient flow.

I went with int6 quantization and zstd-22 to fit 20M parameters into the budget. The wider MLP (3x SwiGLU) uses the headroom well. I train at seqlen 2048 and evaluate with a sliding window at stride 64 so every token gets scored with nearly full context. Weight decay at 0.04 on the Muon optimizer keeps the weight magnitudes low, which makes quantization cleaner. QAT kicks in at 70% through training via straight-through estimation so the model adapts to the int6 rounding before export. SWA averages checkpoints during the warmdown phase for smoother final weights.

I also added a SmearGate at the embedding layer (learned per-dimension blend with the previous token's embedding) and a BigramHash table (XOR-hashed token pair lookup, 4096 buckets) to give the model cheap access to local bigram context without burning attention compute. Both are initialized to near-zero so they learn gradually. Orthogonal initialization on all weight matrices, which I found matters for the gating components to work properly.

The compute tradeoff: 16 effective layers at d=512 with 3x MLP runs about 1.45x slower per step than a standard 11-layer model, landing around 5000 steps in 10 minutes. That is still 2.6B tokens, well above Chinchilla-optimal for 20M parameters.

Configuration: `NUM_LAYERS=8 RECURRENCE=2 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 VOCAB_SIZE=1024 TIE_EMBEDDINGS=1 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 SWA_ENABLED=1 SWA_EVERY=50 QAT_START_FRAC=0.7`

```bash
RUN_ID=depth_rec_8x2 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
