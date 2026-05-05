# PR135 Analysis — Current Best at 1.1539 BPB

## Result: 1.1539 sliding, 1.1748 standard, 15.16MB, 7201 steps @ 83ms

## Novel Techniques (not in any other PR)

### 1. OrthoInit + muP scaling
- All large weight matrices (≥64×64) get `nn.init.orthogonal_(weight, gain=1.0)`
- Output projections (.proj) scaled by `1/sqrt(2*num_layers)`
- "Gives Muon a head start" — better initial conditioning

### 2. BigramHash Embedding
- 4096-bucket hash table, dim=128, projected to 512
- Hash: `XOR(36313 * curr_token, 27191 * prev_token) % 4095`
- Injected into residual stream before first transformer layer
- ~512K extra params but provides bigram-level info cheaply

### 3. SmearGate
- Learned gate (512 params) blending current + previous token embeddings
- `output = (1-sigmoid(gate)) * x + sigmoid(gate) * x_prev`
- Applied after RMSNorm, before transformer blocks

### 4. NO QAT!
- QAT_ENABLED=0 — they DISABLED fake quantization during training
- Reason: "54% step overhead outweighs the quant gap reduction"
- This is opposite to PR128/122/64 which all use QAT
- At 83ms/step they only get 7201 steps — QAT would make it even slower

### 5. seq2048 + batch786K
- TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=786432
- Medium seq length, big batch — gets more steps than seq4096

### 6. Cautious weight decay
- weight_decay=0.01 for AdamW params
- Combined with grad_clip=0.3

## Config
```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3.0 TIE_EMBEDDINGS=1
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128
WEIGHT_DECAY=0.01 QAT_ENABLED=0
EVAL_STRIDE=64
```

## What We Should Try
1. Add SmearGate + BigramHash to our script (~easy, ~0 cost)
2. OrthoInit (~easy, just change _init_weights)
3. Try NO QAT (our QAT adds overhead — maybe hurting step count)
4. Try seq2048 + batch786K instead of seq4096 + batch393K
5. Weight decay=0.01

## Key Insight
PR135 is SLOWER per step (83ms vs our 57ms) but compensates with novel features.
At 83ms they get only 7201 steps. If we can add their features without the speed hit,
we'd get more steps AND better per-step quality.

## Script: experiments/pr135_train_gpt.py
