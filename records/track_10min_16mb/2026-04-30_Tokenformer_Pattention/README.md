# Tokenformer: Pattention yields efficient compression (40% model size)  - first

**Track:** Non-Record, 10-minute 16MB
**Author:** Alex Wu (`@alexwu`)
**Base:** repository root `train_gpt.py` (the simple baseline, not the SOTA stack)

**TLDR:**  
Tokenformer (Pattention) reduces model artifact size by ~40% at matched params, unlocking significant byte-budget margin for future improvements – though with a notable impact on validation BPB (+0.1466 vs. baseline).

**Relevance:**  
1. Novelty: no prior submissions
2. Fit: Tokenformer targets parameter-count efficiency (L(N))
3. SOTA Stack Compatibility: Pattention is readily composable with leading SOTA methods (quant, TTT, etc)

## TokenFormer background

Tokenformer (Wang et al., NeurIPS '24, [arxiv:2410.23168](https://arxiv.org/abs/2410.23168)) replaces every
linear layer `y = x W` in a Transformer with **Pattention**: cross-attention from the input row `x`
to a learnable bank of `P` "parameter tokens." The two halves of the dense weight `W ∈ R^{d_in × d_out}`
become two separate learnable matrices `K ∈ R^{P × d_in}` and `V ∈ R^{P × d_out}`:

```
attn = softmax((x @ K.T) / sqrt(d_in), dim=-1)
y    = attn @ V
```

The paper's motivation is (a) incremental scaling — you can grow capacity by appending
parameter tokens to a pretrained model rather than retraining a wider one — and (b) param count
decoupled from FLOPs in a softer way than dense weights.

For Parameter Golf, the finding is that Pattention is a **parameter-efficient
substitute** for dense `nn.Linear` whenever `P · (d_in + d_out) < d_in · d_out`, i.e. whenever
`P < d_in·d_out / (d_in + d_out)`. Note: I did not test incremental scaling given limited compute - that would be an interesting followup.

## Submission Overview

- v1 scope: replace **only** the MLP `fc` (dim → mlp_mult·dim) and `proj` (mlp_mult·dim → dim) with
  Pattention layers. Attention `q`/`k`/`v`/`proj` stay as `CastedLinear`
- Param-token count is auto-set to **match the dense parameter budget**:
  `P = round(d_in · d_out / (d_in + d_out))`. At the default `MODEL_DIM=512`, `MLP_MULT=2`
  this gives `P = 341` per Pattention. `PATTENTION_P_RATIO=R` lets us scale relative to matched
  for the natural Tokenformer scaling ablation.
- Same activation (`relu^2`) between the two Pattentions; same residual structure; same Muon+Adam
  optimizer split (Pattention K/V are 2D so they auto-route to Muon — no optimizer plumbing needed)
- Same int8+zlib quantization path; same val_bpb pipeline; same warmup/warmdown schedule

The submission `train_gpt.py` is a strict superset of the baseline: `PATTENTION=0` reproduces the
naive baseline exactly; `PATTENTION=1` enables Pattention.

## Reproduction

### Submission run (8xH100 SXM, 10-minute cap)

```bash
RUN_ID=tokenformer_pattention_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
PATTENTION=1 PATTENTION_P_RATIO=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED={7, 42}` for the 3-seed mean.

### Baseline ablation (same command, `PATTENTION=0`)

```bash
RUN_ID=baseline_dense_mlp_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
PATTENTION=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Local MLX

```bash
RUN_ID=mlx_pat_smoke ITERATIONS=20 WARMUP_STEPS=2 \
TRAIN_BATCH_TOKENS=8192 GRAD_ACCUM_STEPS=1 \
VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 PATTENTION=1 \
python3 train_gpt_mlx.py
```

## Local MLX results (1-shard, 20-step smoke)

Identical hyperparameters except `PATTENTION`:

| Metric                | Baseline (`PATTENTION=0`) | Tokenformer (`PATTENTION=1`) |
|----------------------:|--------------------------:|-----------------------------:|
| `model_params`        | 17,059,912                | 17,050,696 (-9k, **matched within 0.05%**) |
| Step 1 train_loss     | 6.9429                    | 6.9429                       |
| Step 5 train_loss     | 9.6199                    | 9.4912                       |
| Step 10 train_loss    | 6.1684                    | 6.2307                       |
| Step 20 train_loss    | 5.8821                    | 6.0181                       |
| ms/step (M1)          | ~1100                     | ~775                         |

Step-1 loss = 6.9429 ≈ ln(1024) = 6.9315 in both runs, confirming the zero-init residual stream
(both `attn.proj.weight` and Pattention `V` are zero-initialized so each block is identity at init).
Both curves decrease monotonically after the same shared step-2 transient (an artifact of
`WARMUP_STEPS=2`, not Pattention-specific).

Full summary logs: [`local_mlx_smoke/mlx_pattention.summary.txt`](local_mlx_smoke/mlx_pattention.summary.txt)
and [`local_mlx_smoke/mlx_baseline.summary.txt`](local_mlx_smoke/mlx_baseline.summary.txt).

## 8xH100 results (10-minute wallclock, single seed=1337, sp1024)

Two head-to-head runs were executed sequentially on a single 8xH100 SXM pod (NVIDIA H100 80GB, torch 2.11.0+cu128, FA3, 80 train shards × ~100M tokens each, val on full 62M-token val split). Wallclock cap = 600s including warmup; submission compute used = ~20 GPU-min total.

| # | Config                            | `model_params` | Steps  | step_avg | pre-quant `val_bpb` | int8+zlib `val_bpb` | Compressed bytes        |
|--:|-----------------------------------|---------------:|-------:|---------:|--------------------:|--------------------:|------------------------:|
| 1 | Baseline `PATTENTION=0`, MLP×2    | 17,059,912     | 11,555 | 51.93 ms |        **1.2231**   |        **1.2305**   | 15,861,681 (15.13 MiB)  |
| 2 | **Pattention matched, MLP×2**     | 17,050,696     | 10,046 | 59.73 ms |          1.3662     |        **1.3771**   |  9,579,528 ( 9.13 MiB)  |
| Δ | (2 − 1)                           | −9,216 (−0.05%)| −1,509 |  +7.8 ms |          +0.1431    |          +0.1466    | **−6,282,153 (−39.6%)** |

**Submission entry** (Run 2, the apples-to-apples matched-params head-to-head): **val_bpb = 1.3771** (int8+zlib roundtrip), **9,579,528 bytes** total submission, 17,050,696 params.

Full per-run training logs: [`logs_8xh100/run1_baseline_pat0.log`](logs_8xh100/run1_baseline_pat0.log), [`logs_8xh100/run2_pat_matched.log`](logs_8xh100/run2_pat_matched.log). Orchestrator: [`logs_8xh100/orchestrator.log`](logs_8xh100/orchestrator.log).

### What the comparison says

1. **Run 1 (baseline control)** replicates the leaderboard's sp1024 naive-baseline area: 1.2305 (int8+zlib) at 11.5k steps is within 0.006 BPB of the 1.2244 reported reference. Confirms the pod, env, dataset, train_gpt.py, and int8+zlib quantization pipeline are all healthy.
2. **Run 2 (matched-params Pattention)** is the apples-to-apples head-to-head. The key finding is the **40% drop in compressed artifact size at matched parameter count**: 15.86 MB → 9.58 MB, a 6.28 MB reduction. Pre-quant model bytes are essentially identical (~67 MB raw FP32 for both, matching the 0.05% param diff). The int8 stage also compresses both runs essentially identically (`payload_ratio` 3.91× → ~17.2 MB int8 raw_torch payload for either run). **The compression win is the zlib stage**: zlib shrinks the dense baseline's int8 payload by only **1.09×** (17.22 MB → 15.81 MB) but shrinks the matched-params Pattention's int8 payload by **1.81×** (17.22 MB → 9.53 MB). Pattention's K/V tables (especially V, which is zero-init and stays low-magnitude / structured for many entries) are dramatically more compressible to zlib's dictionary+Huffman pipeline than dense MLP `fc`/`proj` matrices, which look much closer to uniform noise after training.
3. **val_bpb cost** (1.2305 → 1.3771, +0.147 BPB) is the cost of the swap at this scale. Earlier 1xH100 dev runs at 7× fewer steps showed a +0.124 BPB deficit; the 8xH100 deficit at 10,046 steps is +0.147 BPB, so the gap is not closing with more steps — this is an architectural cost of the matched-params Pattention, not undertraining. (Note: the gap is bigger at *more* steps because the dense baseline is converging faster than Pattention; both are still on a downward trajectory at 600s.)
4. **Quantization is robust to the swap.** `payload_ratio` is 3.91× for both runs. int8+zlib roundtrip drift is small in absolute terms (≤+0.013 BPB on either run).

## Code diff

The Pattention module + MLP swap lives in `train_gpt.py`:

- `class Pattention(nn.Module)`: K (P, d_in) and V (P, d_out) as fp32 master parameters with
  cast-on-matmul (matches `CastedLinear`'s convention so it composes with the existing bf16
  forward and Muon's gradient orthogonalization out of the box).
- `class MLP`: when `pattention_enabled=True`, swaps `fc`/`proj` to Pattention with
  `n_tokens = round(p_ratio · d_in · d_out / (d_in + d_out))` per layer.
- `for module in base_model.modules(): if isinstance(module, Pattention): module.float()` — keeps
  K/V in fp32 master after the bf16 cast, mirroring `CastedLinear`.

The Muon param-grouping in `main()` already keys on `p.ndim == 2`, so K/V are auto-routed without
any other changes:

```
matrix_keys (per block, 8 entries):
  blocks.{i}.attn.c_q.weight  blocks.{i}.attn.c_k.weight  blocks.{i}.attn.c_v.weight
  blocks.{i}.attn.proj.weight blocks.{i}.mlp.fc.K         blocks.{i}.mlp.fc.V
  blocks.{i}.mlp.proj.K       blocks.{i}.mlp.proj.V
```

`train_gpt_mlx.py` carries the analogous diff for local Apple Silicon iteration.

## What to explore next

This submission is meant to be a POC for future exploration. Areas (ranked): 
- **Spending the freed compressed bytes**: deeper-but-narrower stacks, larger vocabularies, or
  more attention heads — anything that natively converts the 6.28 MB of freed artifact budget
  into L(N) at fixed wallclock. (One pilot at `MLP_MULT=4` / 26.5M params landed at val_bpb 1.3746
  in 9,302 steps — essentially identical to the matched-params Pattention; full log preserved at
  [`logs_8xh100/run3_pat_mlp4.log`](logs_8xh100/run3_pat_mlp4.log) for reference. Naive scale-up
  alone does not recover the val gap; it likely needs to be paired with K-sharing or a different
  step/wallclock budget.)
- Combine w/ SOTA stack to see if it plays well
- Incremental scaling: I did not test incremental scaling (paper's original use case) given limited compute - that would be an interesting followup.
- Pattention on `q`/`k`/`v`/`proj` of `CausalSelfAttention`.
- Cross-block parameter-token sharing (one K table per role, shared across all blocks). This is
  the actual parameter-golf angle and the main reason Tokenformer is mechanistically attractive
  for this challenge — saved for a follow-up where the scaling ablation can be run cleanly.
- The `P_RATIO` ablation (P_RATIO ∈ {0.5, 1.0, 2.0}) demonstrating Tokenformer's headline
  "incremental scaling" claim in the parameter-golf setting.

## File manifest

- `train_gpt.py` — the modified PyTorch/GPU script. `PATTENTION=1` is the submission config;
  `PATTENTION=0` reproduces the naive baseline exactly. This is the exact 50,566-byte file that
  produced the 8xH100 logs (matches the `Code size: 50566 bytes` line in `run1_baseline_pat0.log`).
- `train_gpt_mlx.py` — the analogous Pattention diff for local Apple Silicon iteration. The repo
  root keeps the unmodified baseline; this folder ships the Pattention version.
- `local_mlx_smoke/mlx_pattention.summary.txt`, `local_mlx_smoke/mlx_baseline.summary.txt` — local
  20-step smoke summaries from `train_gpt_mlx.py` showing matched params + decreasing loss.
- `requirements.txt` — copied verbatim from repo root.
- `submission.json` — leaderboard metadata.
- `runbook.md` — exact commands for the 1xH100 sanity check and 8xH100 submission runs.
- `logs_8xh100/run1_baseline_pat0.log`, `run2_pat_matched.log` — full per-run 8xH100 training logs for the head-to-head submission comparison (seed=1337, sp1024, MAX_WALLCLOCK_SECONDS=600).
- `logs_8xh100/run3_pat_mlp4.log` — supplementary scale-up pilot (`MLP_MULT=4`, 26.5M params, 11.36 MiB compressed). Not part of the headline comparison; preserved as reference for the v2 "spend the freed bytes" question.
- `logs_8xh100/orchestrator.log` — wallclock for the run sequence.
- `h100_10min_seed1337/baseline_pat0.log`, `tokenformer_pat1.log` — earlier 1xH100 dev runs that motivated the 8xH100 plan (the deficit-doesn't-close-with-more-steps observation came from comparing these two against the 8xH100 numbers).
