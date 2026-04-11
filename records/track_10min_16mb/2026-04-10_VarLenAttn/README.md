# Record: Varlen attention + fused MLP + TTT

**val_loss: 2.7806 | val_bpb: 1.07643** | **~15.99 MB** | 8×H100 SXM, 600s train + ~360s TTT eval
| Seed | SW Loss | SW BPB | TTT Loss | TTT BPB |
|------|---------|--------|----------|---------|
| 0 | 2.78822313 | 1.07940806 | 2.78138792 | 1.07676194 |
| 1 | 2.78698310 | 1.07892801 | 2.78033428 | 1.07635404 |
| 2 | 2.78652675 | 1.07875134 | 2.77993034 | 1.07619767 |
| **Mean** | **2.78724** | **1.07903** | **2.78055** | **1.07644** |

Best PR bpb ([PR #1523](https://github.com/openai/parameter-golf/pull/1523)): 1.0778. **delta=.0014**
Merged record bpb ([PR #1493](https://github.com/openai/parameter-golf/pull/1493)): 1.0810. **delta=.0047**

Increased training speed ~5% via variable length attention, a fused kernel, and grouping together small parameters, yielding ~.002 nats when comparing sliding window eval. Re-added document-based LoRA TTT which has *no inter-sequence dependence* and beats previous record by ~.003 nats.

## Main changes

Applied changes from [my old PR](https://github.com/openai/parameter-golf/pull/1354) to a recent record PR: [Record: SP8192 + Triple Recurrence + Banking + Fused MLP + Muon 0.97 — val_bpb 1.0778 (3-seed mean) #1523](https://github.com/openai/parameter-golf/pull/1523). Most of below is copied from my previous PR.

This involves 3 things:

### 1. Variable length attention (~2% faster training, ~0.001 nats)

Replaced dense causal attention with Flash Attention 3's `flash_attn_varlen_func`. During training, documents are packed into flat token buffers with `cu_seqlens` boundaries so attention is computed within documents only — the model never attends across unrelated documents that happen to be adjacent in a batch.

This does two things:
- Removes the need for the model to learn to ignore pre-BOS content from unrelated documents
- Reduces wasted FLOPs: e.g. 10 short (100-token) docs packed into a 1k-token buffer cost proportional to `100 * 100**2 = 1M` attention FLOPs vs `10 * 1000**2 = 10M` with dense attention.

### 2. Fused MLP + grouped small params (~3% faster training, ~0.001 nats)

A custom Triton kernel (`linear_leaky_relu_square_kernel`) fuses the up-projection, LeakyReLU(0.5)² activation, and squaring into a single kernel. Based on similar kernels from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/triton_kernels.py). I also group the many tiny replicated scalar/control gradients into a single all-reduce to avoid a pile of tiny collectives.

### 3. Doc-based test-time training (TTT) (~0.003 nats)

> [Blog explaining LoRA-based TTT from past record](https://samacquaviva.com/projects/parameter-golf/)

Although it is technically legal in this competition to train on tokens from previous documents in the dataset, I am spiritually opposed to this. Under the current formulation, if the eval set was bigger, the expectation of the loss would be lower which seems broken. So in this implementation, there is score-first TTT applied to each sequence in the validation set *independently* (and efficiently using batched LoRAs), which is strictly harder.

Re-adds LoRA-based TTT, based on [my old implementation](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md), but > 2x faster which allows for using smaller chunk sizes which leads to better performance. This is an instance of "Case 3" according to [this classification](https://samacquaviva.com/projects/ttt-clarification/). The TTT gain is only ~.003 nats over sliding window evaluation, as compared to ~.007 in [my previous PR](https://github.com/openai/parameter-golf/pull/1354) and ~.006 in the most recent record which trains on the entire validation sequence. This gap is probably closeable with better hparams, I largely just took my hparams optimized for the non-looped transformer.

It's interesting to note that adding test-time training improves loss more than adding ~215 steps. These 215 steps train on `786432*215=169,082,880` tokens to gain ~.002 nats. The average sequence length in the validation set is ~200 tokens which means test-time training here gains ~.003 nats / 800 tokens on average (valid bc sequences are trained independently). So, in a way, TTT is `~(.003/800) / (.002/169082880) >= 300k` times more token efficient than pre-training: it helps to be in distribution :)

## Other small changes

- Added some useful dev things, like loading from a checkpoint just for eval
- Didn't submit minified code, instead wrote that utility into the script so that it is easier for people to build off of this

## Replicating runs + dev

```bash
# setup
uv venv
source .venv/bin/activate
uv pip install -r records/track_10min_16mb/2026-04-10_VarLenAttn/requirements.txt
uv pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
uv pip install torch==2.9.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards  128

# train + eval
SEED=0
ARTIFACT_DIR="runs/varlen${SEED}" SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-10_VarLenAttn/train_gpt.py

# eval saved checkpoint w/ TTT (useful for dev)
EVAL_ONLY_PATH="runs/varlen${SEED}/final_model.pt" SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-10_VarLenAttn/train_gpt.py
```
