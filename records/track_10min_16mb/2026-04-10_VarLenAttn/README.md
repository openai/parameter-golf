# Record: Varlen attention + fused MLP + TTT

**val_loss: 2.77261 | val_bpb: 1.07336** | **~15.99 MB** | 8×H100 SXM, 587s train + ~340s TTT eval
| Seed | BPB | Loss |
|------|-----|------|
| 0 | 1.07258208 | 2.77059090 |
| 1 | 1.07324696 | 2.77230836 |
| 2 | 1.07426259 | 2.77493185 |
| **Mean** | **1.07336388** | **2.77261037** |
| **Std** | **0.00084633** | **0.00218618** |

Best PR bpb ([PR #1529](https://github.com/openai/parameter-golf/pull/1529)): bpb=1.0753 (**delta=0.0019**), loss=2.7776 (**delta=0.0050**)

Merged record bpb ([PR #1493](https://github.com/openai/parameter-golf/pull/1493)): bpb=1.0810 (**delta=0.0076**), loss=2.7923 (**delta=0.0197**)

Increased training speed ~5% via variable length attention, a fused MLP triton kernel (no `cutlass_evt_fusion` dep), and grouping together small parameters, yielding ~.002 nats when comparing sliding window eval. Re-added document-based LoRA TTT which has *no inter-sequence dependence* and improves over strided evaluation by ~.008 nats.

## Main changes

Applied changes from [my old PR](https://github.com/openai/parameter-golf/pull/1354) to a recent record PR: [#1523](https://github.com/openai/parameter-golf/pull/1523). But [PR #1552](https://github.com/openai/parameter-golf/pull/1552) beat my previous bpb before I submitted the PR, so I incorporated their (orthogonal) improvements. Most of below is copied from my previous PR [#1354](https://github.com/openai/parameter-golf/pull/1354).

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

Re-adds LoRA-based TTT, based on [my old implementation](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md), but > 2x faster which allows for using smaller chunk sizes which leads to better performance. This is an instance of "Case 3" according to [this classification](https://samacquaviva.com/projects/ttt-clarification/).

It's interesting to note that adding test-time training improves loss more than adding ~215 steps. These 215 steps train on `786432*215=169,082,880` tokens to gain ~.002 nats. The average sequence length in the validation set is ~200 tokens which means test-time training here gains ~.003 nats / 800 tokens on average (valid bc sequences are trained independently). So, in a way, TTT is `~(.003/800) / (.002/169082880) >= 300k` times more token efficient than pre-training: it helps to be in distribution :)

## Other small changes

Made some changes to make replication and dev based on this PR easier:

- Load from a checkpoint just for eval
- Didn't submit minified code, instead wrote that utility into the script when calculating file size so that it is easier for people to build off of this
- Store unminified code in logs

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
