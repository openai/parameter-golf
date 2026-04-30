# RT-KV PR2014 Experiment

This branch tests a Recurrent Transformer-style key/value recurrence overlay on top of the PR #2014 Parameter Golf stack.

The idea is motivated by [The Recurrent Transformer: Greater Effective Depth and Efficient Decoding](https://arxiv.org/abs/2604.21215). The paper argues that standard Transformers are temporally shallow because each position can only attend to key/value states computed by previous layers. A Recurrent Transformer instead lets a layer attend to key/value states derived from its own activations, increasing effective depth while keeping autoregressive decoding cost practical. It also describes an exact tiled training/prefill algorithm intended to avoid the naive bandwidth cost of revealing recurrent keys and values sequentially.

## What We Are Trying

The experiment keeps the PR #2014 CaseOps/SP8192 training, quantization, compression, and score-first TTT setup, then enables an RT-KV overlay in `train_gpt_RT.py`:

- `RT_KV_ENABLED=1`
- `RT_KV_START=4`
- `RT_KV_END=4`
- `RT_KV_MIN_LOOP_PASS=2`
- `RT_KV_FAST_APPROX=1`

In plain terms, we are testing whether adding recurrent key/value behavior to one looped layer can improve validation BPB without breaking the 10 minute training and 10 minute eval budgets. The first run uses seed `42`; if it is promising, logs can be added one by one for seeds `314` and `0`.

## PR2014 Base

This branch starts from PR #2014, which reported `val_bpb=1.05759` as a 3-seed mean. The base stack includes:

- SP8192 CaseOps data with original-byte validation sidecars.
- Progressive training context growth: `1024@0.100,2048@0.700,3072@1.000`.
- Final/eval/TTT context at 3072 tokens with `EVAL_STRIDE=1536`.
- Quantized phased LoRA TTT with `TTT_MASK=no_qv`.
- Short-document score-first TTT chunks: `TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24`.
- LQER asymmetric rank-4 correction, AWQ-lite, asymmetric logit rescale, GPTQ int6 matrices, int7 embeddings, and per-group `lrzip` compression.

## Lineage And Credits

This experiment is intentionally a small change on top of the public PR #2014 lineage:

- PR #2014 by @simonbissonnette: progressive 3k context growth plus short-doc score-first TTT.
- PR #1855 by @codemath3000: merged CaseOps/SP8192/LQER/SparseAttnGate/BOS-fixed SmearGate record baseline.
- PR #1953: long-context/no_qv TTT mask and `QK_GAIN_INIT=5.25` sweep lineage.
- PR #1945, PR #1908, and PR #1923: late-April quantization stack, including AWQ-lite and asymmetric logit rescale.
- PR #1797 by @dexhunter: SmearGate and LQER asymmetric rank-4 lineage.
- PR #1787 by @nprime06: Polar Express Muon, `MIN_LR`, SparseAttnGate, and fused CE lineage.
- PR #1736 and PR #1729 by @dexhunter / @romeerp: CaseOps integration and byte sidecar accounting.
- PR #1667 by @MarioPaerle: SmearGate lineage.
- PR #1626 and PR #1610: phased score-first TTT lineage.
- Issue #1017 by @cocohearts: score-first validation criteria.

## Current Run Command

```bash
RT_KV_ENABLED=1 \
RT_KV_START=4 \
RT_KV_END=4 \
RT_KV_MIN_LOOP_PASS=2 \
RT_KV_FAST_APPROX=1 \
torchrun --standalone --nproc_per_node=8 train_gpt_RT.py
```

For leaderboard-style runs, use the full PR #2014 environment from the record README as well, including CaseOps data paths, 3072-token context settings, quantization settings, and TTT settings.
