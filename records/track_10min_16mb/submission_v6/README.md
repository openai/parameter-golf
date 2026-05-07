# non-record submission v6 — request for evaluation

stack of small additive deltas on top of merged PR #2014 (val_bpb 1.05759, 3-seed). every change is well-isolated and zero/near-zero in artifact cost. **could not afford the full 8×H100 run to confirm the number** — explained below. asking maintainers to run a single seed if possible.

## the situation, honestly

i'm a solo participant working from spain on personal credit. i applied multiple times for the compute credits the competition was offering and didn't get a response. with the deadline closing today, i pooled some money and rented a pod for the final stretch — got the code uploaded, kicked off seed 42, and the pod's TCP gateway dropped before the run finished. i don't have the budget for another full attempt and pod time runs out at the deadline regardless.

so this PR is the code, fully self-contained and ready to run, plus an honest prediction. if there's any chance someone with infra can reproduce it i'd be incredibly grateful — i think the result is competitive and the changes are interesting independent of whether they help me personally.

## predicted val_bpb: ~1.052 (optimistic) to ~1.056 (conservative)

base PR #2014 = 1.05759 (3-seed mean, std 0.00034, confirmed in their logs). my deltas:

| change | expected delta | source / confidence |
|---|---|---|
| gated XSA (per-head tanh-alpha, zero-init) | -0.001 to -0.003 | modded-nanogpt PR #264, p=0.0014. zero-init means step-0 is bit-identical to baseline so this is strictly additive in expectation |
| LeakyReLU² slope 0.5 → 0.3 | -0.0007 | sweep result from PR #1948, isolated |
| GPTQ all-rank Hessian averaging | -0.0001 to -0.0005 | reduces per-rank calibration noise; small but measurable |
| reverse-cholesky `Hinv` | 0 BPB direct, +3-5s training budget back | algorithmic identity, ~2× faster than `cholesky_inverse` + re-cholesky |

**summed expectation: -0.002 to -0.005 BPB → val_bpb ≈ 1.053–1.056, optimistic ~1.052.**

i'd guess it lands around **1.054** on a 3-seed mean. that would put it competitive with the current top of the legit cluster.

## what's in the stack

base = PR #2014 file, untouched except for these surgical edits to `train_gpt.py`:

1. **gated XSA**: 5 small edits in `CausalSelfAttention` / `Block` / `GPT.__init__`. adds one fp32 vector `xsa_alpha` of shape `(num_heads,)` per layer (~16 bytes/layer artifact cost). `tanh(alpha)` factor multiplies the existing XSA subtraction coefficient. ndim=1 so it routes to the scalar AdamW group automatically.

2. **leaky-relu² slope 0.3**: changed in both the python path AND the fused triton kernel (lines 1016/1017 backward derivative `2·0.3²·c = 0.18·c`, lines 1021/1022 forward `0.3·c`). without patching the kernel you get a train/eval mismatch.

3. **GPTQ all-rank Hessian averaging**: hparam `gptq_all_reduce` (default 1), 8 lines around `return hessians` to do `dist.all_reduce` across world_size before normalizing.

4. **reverse-cholesky Hinv**: 6-line swap in `gptq_quantize_weight`. `H_flip = flip(H, (0,1)); L = cholesky(H_flip); U = flip(L, (0,1)); Hinv = solve_triangular(U, eye, upper=True)`. mathematically identical to the original 3-op pipeline, but ~2× faster.

env knobs default ON: `GATED_XSA=1`, `GPTQ_ALL_REDUCE=1`. everything else inherits from PR #2014.

## why it should compose well

every delta is independently additive:
- gated XSA is zero-init → step-0 is the baseline model, only learnable improvement
- slope 0.3 narrows the negative side, lowers post-quant degradation
- all-rank Hessian smooths out a known noise source
- reverse-cholesky is pure algorithmic speedup

no overlapping mechanism between them. should compose linearly within the noise floor.

## reproducing

```bash
bash launch.sh   # default SEED=42, override SEED=314 / SEED=0
```

needs:
- 8×H100 80GB SXM
- pytorch 2.9.1+cu128, triton 3.5+, FA3
- system `lrzip` binary
- the standard caseops sp8192 fineweb shards + the included tokenizer

`launch.sh` has the full env block (identical to PR #2014's reproducer + the two new flags).

## ask

please run 1 seed if there's compute available. i'd love to know if the prediction lands. happy to coordinate, answer questions, or simplify anything if it helps a reviewer.

thank you for putting on the competition — it's been an incredible learning experience even without making the final scoreboard.

## files

- `train_gpt.py` — main script (4569 lines, +35 vs PR #2014 base)
- `launch.sh` — env vars + torchrun
- `lossless_caps.py`, `prepare_caseops_data.py` — caseops data prep helpers (untouched from base)
- `tokenizers/` — sp8192 caseops bpe model (untouched from base)
- `requirements.txt`
- `submission.json` — metadata, `val_bpb: null` (no completed run)
