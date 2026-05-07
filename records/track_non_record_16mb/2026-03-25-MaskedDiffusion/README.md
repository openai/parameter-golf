This is a non-record submission that replaces the autoregressive `train_gpt.py` baseline with a masked diffusion language model implemented in`train_mdlm.py`. The MDLM is from ["Simple and Effective Masked Diffusion Language Models"](https://arxiv.org/pdf/2406.07524), and the code inspired by ["that paper's repo"](https://github.com/kuleshov-group/mdlm)

The model keeps much of the original training stack from the original baseline, but swaps the causal next-token objective for a bidirectional masked denoising objective with iterative sampling. Since the addition of the conditioning weights pushes us over the 16MB limit, I adopt the common int6+int8 mixed quantization, zstd-22 compression strategy from (#) so both models have 9 layers.

## Config
- Tokenizer/data: reuses FineWeb SP-1024, one extra \[MASK\] token added for 1025 vocab size
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`; Identical to baseline
- Dropout 0: Arguably less important in diffusion models since they already handle a lot of noise, which acts as regularization in the same way dropout does.
- Attention: bidirectional transformer with GQA-style `NUM_KV_HEADS=4`, with adaLN conditioning
- Conditioning: timestep-conditioned denoiser with reduced internal conditioning width `cond_dim=max(model_dim//4, 64)`
- Batch/sequence defaults: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=256`. Lower sequence length because it's a bidirectional model
- Sampling defaults: `SAMPLER=ddpm_cache SAMPLING_SCHEDULE=linear SAMPLING_STEPS=256`. N.b. probably lots of fun to be had with the sampling schedule!
- Variational eval: `VAR_EVAL_STEPS=32`. I'm interested in whether using more val steps gives better performance, which would be a kind of test-time compute. Validation takes ~2min on 8XH100. Running at `VAR_EVAL_STEPS=128` gets our upper bound down to 1.59, but at the cost of taking 8 mins to evaluate, which is an invalid submission.

## Metrics
- `val_loss` is the continuous-time SUBS denoising objective used for training.
- `val_var_bpb` is the compression-facing metric for this folder. It is a byte-normalized variational upper bound on NLL obtained by discretizing the same absorbing-mask process at evaluation time.

### Variational BPB

The variational BPB reported here is not apples-to-apples comparable with the validation BPB from the autoregressive models, which means this is a particularly special non-record submission. The variational metric was added under duress because there is no perfect analogy to autoregressive models' losses in the diffusion regime:
- A masked diffusion model does not natively provide an exact autoregressive factorization of `p(x)` token-by-token, so the training loss is not directly convertible to an exact codelength.
- Obtaining the exact codelength for the continuous-time process would require integrating over latent corruption trajectories, which is not tractable in our eval time.
- To make compression more comparable with AR baselines, eval instead reports a discrete absorbing-mask variational bound:
  - terminal KL term `KL(q(x_T | x_0) || p(x_T))`
  - plus a sum of reverse-process KL terms across `VAR_EVAL_STEPS`
- This is still an upper bound rather than an exact BPB, but it is much more principled than simply converting the denoising loss into BPB units, as if it is analogous to CE.
- This also allows us to measure the impact of discretization on the model as a form of test-time compute by varying `VAR_EVAL_STEPS`, which I note anecdotally has a meaningful impact on the metric - running at `VAR_EVAL_STEPS=128` gets our upper bound down to 1.59, but at the cost of taking 8 mins to evaluate.

Command:
```bash
RUN_ID=baseline_mdlm DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VAL_LOSS_EVERY=0 VAR_EVAL_STEPS=32 COND_DIM=128 torchrun --standalone --nproc_per_node=8 records/track_non_record_16mb/2026-03-25-MaskedDiffusion/train_mdlm.py
```

Recommended knobs to play with:
- `TRAIN_SEQ_LEN`: diffusion currently defaults to `512` rather than the AR baseline's `1024` because shorter windows improve throughput and increase the number of independent timestep samples per step, and in theory should not harm diffusion models as much.
- `SAMPLING_STEPS` / `SAMPLING_SCHEDULE`: test-time compute knobs for generation; they do not affect `val_var_bpb` but could be cool to play with and visualize
- `VAR_EVAL_STEPS`: tighter but slower variational evaluation.

Things that really didn't work:
- This is the best method of a bad crop of diffusion LM methods; I have implemented a continuous diffusion model a la DiffusionLM, I may push it later, but it sucks even more
- Don't attempt to tie the weights - the loss spikes, they're not doing a symmetric task.

This folder is a proof-of-concept diffusion adaptation rather than a final tuned submission. With enough work, this could plausibly compete with the very worst autoregressive approaches. I don't care to do that, though, because I don't really find diffusion LMs that cool.

Files in this folder:
- `train_mdlm.py` - single-file masked diffusion training/eval script
- `train.log` - training log on Hyperbolic 8xH100
- `submission.json`
- `README.md`

Metrics:
- best pre-quant `val_loss`: 2.6564   (step:13818/20000)
- best pre-quant `val_var_bpb`: 1.6259
- post-quant roundtrip `val_loss`: 2.6553
- post-quant roundtrip `val_var_bpb`: 1.6252
- final_quant_roundtrip_exact `val_loss`: 2.65530960
- final_quant_roundtrip_exact `val_var_bpb`: 1.62519980
- step time / wallclock: 600032 ms total for 13818 steps (step avg: 43.42 ms)
- compressed artifact size: 15,313,980 bytes (int6+zstd22, payload: 21,537,286, raw_torch: 21,589,365, payload_ratio: 3.91x)
- total submission size int6+zstd22: 15,379,114 bytes

