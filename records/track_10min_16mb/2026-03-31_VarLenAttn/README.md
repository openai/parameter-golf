# Record: Varlen attention + fused MLP + TTT

**val_loss: 1.8728 | val_bpb: 1.1092** | **~15.9 MB** | 8×H100 SXM, 600s train + ~420s TTT eval

*Note: this record is WIP because 1). code/logs need to be cleaned up, and 2). if you count quantization, it is ~13' for eval (3-4' for generating calibration data, ~4' for quantization, ~5' TTT). Speeding this up is very doable, and I will update this PR with logs and cleaned up code hopefully today, but wanted to get results in before because this is based on work from a hackathon last Sunday :)*

## Main changes

Improves upon record [2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md) with 3 things:

### 1. Variable length attention (~1% faster training, ~0.001 nats)

Replaced dense causal attention with Flash Attention 3's `flash_attn_varlen_func`. During training, documents are packed into flat token buffers with `cu_seqlens` boundaries so attention is computed within documents only — the model never attends across unrelated documents that happen to be adjacent in a batch.

This does two things:
- Removes the need for the model to learn to ignore pre-BOS content from unrelated documents (this actually doesn't seem to be a major effect, at least in the current implementation, per-step loss is not noticeably better).
- Reduces wasted FLOPs: e.g. 10 short (100-token) docs packed into a 1k-token buffer cost proportional to `100 * 100**2 = 1M` attention FLOPs vs `10 * 1000**2 = 10M` with dense attention. This leads to ~1% faster training on 8xH100, and the additional training steps buy ~0.001 nats improvement. This improvement is limited because the model is so small that there is a lot of overhead which is not in the attention so it can only be sped up so much.

### 2. Fused MLP (~1% faster training, ~0.001 nats)

A custom Triton kernel (`linear_leaky_relu_square_kernel`) fuses the up-projection, LeakyReLU(0.5)² activation, and squaring into a single kernel. Based on similar kernels from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/triton_kernels.py). Also ~1% faster training on 8xH100, yielding another ~0.001 nats improvement.

### 3. Test-time training (TTT) (~0.007 nats)

> [Blog explaining LoRA-based TTT from past record](https://samacquaviva.com/projects/parameter-golf/)

Re-adds LoRA-based TTT, based on [my old implementation](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md) which buys **~0.007 nats**. This is an instance of "Case 3" according to [this classification](https://samacquaviva.com/projects/ttt-clarification/). In the previous record, TTT had a mismatch with train: sequences did not attend to other sequences during TTT/eval but did during training. Here, now that we are not attending to other sequences during training, this is avoided (> 70% of training sequences in the old code attended to previous sequences), compared to ~0% now). The bigger improvement comes from using a *smaller chunk size* (32 instead of 256, so more gradient updates per sequence) and *using RMSProp instead of Adam* (just set `beta1=0` for Adam during TTT). To be able to use a smaller chunk size, I had to substantially optimize the TTT speed via vectorizing ops and better batch scheduling system, ~2x'ing the old implementation's speed.

#### TTT analysis

As you can see below, TTT helps the most at later positions in the document (note this is run on 1/10th of the validation set). The top plot is showing, at each position x in a sequence:
1. What % of tokens in the dataset are at later positions?
2. What % of the gain from test-time training comes from tokens at later positions?

Even though only ~5% of the tokens in the dataset are at postion 10k or later in their sequence, > 15% of the loss improvement from TTT comes from those later positions. In the bottom plot, you can see that if we only cared about long-context performance and only looked at positions 10k and up, the gain from TTT would be much greater than 0.01 nats!

![TTT gain](ttt-gain.png)

## Run results

```bash
sam:~/parameter-golf# python records/track_10min_16mb/2026-03-31_VarLenAttn/calc_p.py \
    --logs records/track_10min_16mb/2026-03-31_VarLenAttn/seed1-eval.txt \
        records/track_10min_16mb/2026-03-31_VarLenAttn/seed2-eval.txt \
        records/track_10min_16mb/2026-03-31_VarLenAttn/seed1337-total.txt
baseline val_loss: [1.88276292 1.88156874 1.88220393]  mean=1.882179
new      val_loss: [1.87311066 1.87298163 1.87220704]  mean=1.872766
delta (baseline - new): 0.009412

baseline val_bpb:  [1.1150812  1.11437394 1.11475014]  mean=1.114735
new      val_bpb:  [1.10936157 1.10928515 1.1088264 ]  mean=1.109158
delta (baseline - new): 0.005577

val delta loss threshold: 0.005
p-value (new is ≥0.005 below baseline): 0.000353
```

Also note that the logs for this run are 5 files, not 3. For seeds 1 and 2, I ran training before implementing/tuning TTT, so to save compute I did not re-run training, but just loaded the checkpoint. For clarity, I will re-run with the final code hopefully later today.

## Replicating runs + dev

```bash
# setup
uv venv
source .venv/bin/activate
uv pip install -r records/track_10min_16mb/2026-03-31_VarLenAttn/requirements.txt
uv pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
uv pip install torch==2.9.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# download data
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# train + eval
SEED=0
ARTIFACT_DIR="runs/varlen${SEED}" SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-31_VarLenAttn/train_gpt.py

# eval saved checkpoint w/ TTT (useful for dev)
EVAL_ONLY_PATH="runs/varlen${SEED}/final_model.int6.ptz" SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-31_VarLenAttn/train_gpt.py
```
