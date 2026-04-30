# Non-record: #1610 reproduction (Δ=+1.9e-5 BPB), n-gram posterior corrector negative result, quantized-eval-only path fix

This folder contains a non-record evidence package built around three separable contributions. It is not a leaderboard claim.

## Prior context

Previous submissions in this line: [#1101](https://github.com/openai/parameter-golf/pull/1101) (pre-TTT anchor, 1.1290 BPB), [#1307](https://github.com/openai/parameter-golf/pull/1307) (07c1 strict base proof vs merged #1019), [#1598](https://github.com/openai/parameter-golf/pull/1598) (SP8192-D 5-seed evidence package).

## Contributions

1. **Reproduction of PR #1610 on independent infrastructure.** Seed-0 BPB is 1.07218477, which differs from #1610's published seed-0 number 1.07216564 by +1.913×10⁻⁵ BPB. Training was run on 8× NVIDIA H100 80GB HBM3 SXM5 on RunPod at this branch's commit `1765afc`, which pins PR #1610 at its upstream commit `ca19195`.
2. **Bounded negative result for a score-first n-gram posterior corrector** layered on PR #1610's phased LoRA TTT eval path. All three tested `(alpha, orders)` configurations degrade BPB relative to the reproduced baseline. Damage scales monotonically with the blend weight `alpha`. Multi-order backoff provides no measurable benefit over single-order at the same `alpha` in this grid.
3. **Bug fix in the quantized-eval-only branch of `train_gpt.py`.** The pre-quantization diagnostic eval previously ran unconditionally, dereferencing `compiled_model = None` in `eval_only_quantized_path` mode. The fix wraps the diagnostic in `if not quantized_eval_only:` (line 3204) and extends the `del eval_model, compiled_model` cleanup guard to cover the same branch (line 3259). Without these two guards, `EVAL_ONLY_QUANTIZED_PATH` ablations could not run. They produced the measurements in Contribution 2.

The reproduction is a credibility prerequisite for the negative-result claim, not a contribution in itself. The corrector formulation and its Section-III-compliance engineering are the only novel content in this package. The bug fix is incidental — surfaced while running Contribution 2.

## Reproduction result

| | Value |
|---|---|
| Our seed-0 BPB | 1.07218477 |
| PR #1610 published seed-0 BPB | 1.07216564 |
| Δ vs published seed-0 | **+1.913×10⁻⁵** |
| Eval wall-clock (s) | 455.9 |
| Artifact bytes (model + code) | 15,999,394 |

Training stopped at step 4,879 of 20,000 because of `MAX_WALLCLOCK_SECONDS=600` (`minus GPTQ_RESERVE_SECONDS=13`). This is the by-design behavior of #1610. The artifact at 15,999,394 bytes leaves 606 bytes of competition headroom; our internal pipeline's stricter 15,997,520-byte threshold (intended to absorb code-size drift between sessions) is the source of the `GATE_A: FAIL` line in the log tail. The full run log is in `train_seed0.log`. Machine-readable summary is in `reproduction_summary.json`.

## Corrector ablation

The corrector is a backward-looking posterior over the scored prefix. At position `t` it maintains a Laplace-smoothed unigram distribution `q_uni(v)` and, for the requested n-gram orders, conditional counts `q_ngram(v | ctx)`. The blend adds `alpha * log(q_t(v))` to the neural logits before scoring, where `q_t(v) = q_ngram(v | ctx_t)` when an n-gram hit exists at order `n` and `q_t(v) = q_uni(v)` otherwise (Laplace guarantees `q_uni(v) > 0` for all `v`). The corrector integrates into the phased TTT loop at `forward_ttt(..., logit_bias=...)` (`train_gpt.py:1048`); the actual blend is the single line `logits = logits + logit_bias` (`train_gpt.py:1122`).

All three ablations were run in eval-only mode against the seed-0 checkpoint from the reproduction above (no retraining).

| Run | alpha | orders | BPB | Δ BPB (run − baseline; positive = worse) | Eval (s) |
|---|---:|---|---:|---:|---:|
| Baseline (no corrector) | 0.0 | — | 1.07218477 | 0 | 455.9 |
| Ablation 1a | 0.3 | [8] | 1.08876294 | **+0.01658** | 462.8 |
| Ablation 1b | 0.3 | [5, 8, 12] | 1.08891256 | **+0.01673** | 472.4 |
| Ablation 1c | 0.1 | [5, 8, 12] | 1.07430360 | **+0.00212** | 465.8 |

**Interpretation.** The corrector's effect at `alpha = 0.1` is approximately 1/8 of its effect at `alpha = 0.3`, consistent with first-order linearity in `alpha` and inconsistent with any threshold-activated improvement at lower blend weights. Multi-order backoff at the same `alpha` produced a negligible delta (`+0.01658` for `[8]` vs `+0.01673` for `[5, 8, 12]`).

Structurally, TTT-LoRA adapts the base model's output distribution using the same scored prefix `x_{1..t-1}` that feeds the corrector's n-gram tables. Both signals are therefore deterministic functions of that prefix. Adding `alpha * log(q_prefix_ngram(v))` on top of logits that already encode `P(x_t | x_{1..t-1})` under TTT adaptation over-counts the prefix evidence; the corrector's positive coefficient systematically pushes probability mass toward tokens the base model has already concluded are likely. This predicts the result is monotonic in `alpha` and that a corrector layered on a non-adaptive (non-TTT) eval pipeline would behave differently. The latter was not tested in this package.

**This PR rules out one tested posterior-corrector path on a reproduced #1610-class phased-TTT stack; it does not claim that all n-gram or posterior correctors are ineffective.**

Raw eval logs are in `ablation_1a.log`, `ablation_1b.log`, `ablation_1c.log`. Machine-readable config + results in `ablation_summary.json`.

## Eval-only bug fix

Two `train_gpt.py` branches went through `base_model = None; compiled_model = None; compiled_forward_logits = None` in `EVAL_ONLY_QUANTIZED_PATH` mode (line 3188) but were then used by downstream eval code:

- The pre-quantization diagnostic `timed_eval("diagnostic pre-quantization post-ema", eval_val, ..., compiled_model, ...)` dereferenced `compiled_model.forward_logits` and crashed all ranks with `AttributeError: 'NoneType' ...`.
- The subsequent `del eval_model, compiled_model` cleanup in the TTT branch referenced `eval_model` which was never bound in this mode, raising `UnboundLocalError`.

The fix adds two guards:

```
# train_gpt.py:3204
if not quantized_eval_only:
    timed_eval(
        "diagnostic pre-quantization post-ema",
        eval_val, h, device, val_data,
        compiled_model, compiled_forward_logits,
    )

# train_gpt.py:3259
if h.ttt_enabled:
    if not ttt_only_eval and not quantized_eval_only:
        del eval_model, compiled_model
```

The post-quantization diagnostic still runs in this branch because it calls `deserialize(h, device)` directly and does not touch the `None` locals.

## Compliance with Issue #1017 Section III

Each of the four conditions walked against the as-shipped corrector code in `train_gpt.py`.

**Condition 1 (strict causal dependence).** State is the `PrefixNgramCorrector` instance defined at `train_gpt.py:15-58`. Its `get_logit_bias()` reads only `self.hist` and `self.uni`, which are populated exclusively inside `update()`. `update(x_t)` is called in the eval loop *after* `F.cross_entropy` scores position `t`. Nothing else writes into `self.hist` or `self.uni`. No future tokens, no external data.

**Condition 2 (full normalized distribution).** The blend is `final_logits[v] = neural_logits[v] + alpha * log(q_t(v))` over the full `V = vocab_size` alphabet (`train_gpt.py:1122`, `[V]` tensor add — not a gathered single-index update). Laplace smoothing at init (`self.uni = torch.ones(V, dtype=torch.int32)` at line 23) guarantees `q_uni(v) > 0` for every `v ∈ V`, so `log(q_t(v))` is finite everywhere and `softmax(final_logits)` is a valid distribution. The `[V]` bias vector is `alpha * (self._lu - self._lz)` at line 34, with the n-gram delta sparsely added over tokens that have n-gram hits; the final effective bias is `alpha * log(q_ngram)` where n-gram hits exist and `alpha * log(q_uni)` otherwise. No dense `[batch, seq, vocab]` allocation exists on the production path; the `[B, 1, V]` bias is broadcast at `train_gpt.py:2565`.

**Condition 3 (score-before-update).** In the phased TTT eval loop (`train_gpt.py:2562-2598`): the bias is collected from `[correctors[_b].get_logit_bias() ...]` (line 2564) *before* the scoring forward pass `forward_ttt_train(..., logit_bias=_logit_bias)` (line 2567). Scores are accumulated in `_accumulate_bpb(...)` (lines 2568-2582). Only then does the update path run: `correctors[_b].update(_tok)` (line 2591), inside the block introduced by the explicit comment `# Corrector: update state with scored tokens (score-before-update)` (line 2583). `PrefixNgramCorrector`'s docstring (lines 17-18) encodes the same contract and the call sites honor it.

Note on the chunk-static approximation. The bias is computed once per TTT chunk (chunk size = `h.ttt_chunk_size = 32`) and broadcast as `[B, 1, V]` to every position inside that chunk. This is a deliberate engineering choice: a per-position bias would require either 32× as many GPU forward passes across the eval set (blowing past the 600 s budget by an unrecoverable margin) or a dense `[B, S, V]` correction tensor with `B=64`, `S=2048`, `V=8192` in bf16 ≈ 2.1 GB per batch per rank — unusable inside the #1610 memory envelope once stacked with activations and optimizer state. The trade-off is accepted because the bias at any position inside chunk `c` is still a function only of tokens from chunks `[0, c)`, which preserves score-before-update at chunk granularity. Within a chunk, the bias is constant — it does not use tokens from the current chunk for its own scoring. This satisfies score-before-update at chunk granularity rather than per-position, and the choice is explicit in the corrector's docstring.

**Condition 4 (single left-to-right pass).** `eval_val_ttt_phased` is one forward pass over the validation token stream. No re-scoring, no second pass, no min-over-runs selection. The phased TTT loop performs interleaved global SGD steps on the base model between chunks, but those SGD steps do not re-score previously scored positions. After each global SGD step, the corrector state is reset (`correctors[_b].reset()`) to avoid stale counts against the updated base model.

**Warmup uses synthetic tokens only.** The TTT compile warmup (`train_gpt.py:3324-3365`) is bracketed by `# BEGIN warmup synthetic tokens` / `# END warmup synthetic tokens` comments and uses a device-local generator (`_warmup_gen = torch.Generator(device=device).manual_seed(0)`) that does not mutate global RNG state. Tokens are drawn via `torch.randint(0, h.vocab_size, ..., generator=_warmup_gen)`. When `corrector_alpha > 0`, a second warmup pass with `dummy_bias = torch.zeros(bsz, 1, h.vocab_size, ...)` precompiles the `logit_bias` branch so Dynamo does not recompile inside the eval timer. The timer starts at `torch.cuda.synchronize(); t_ttt = time.perf_counter()` (`train_gpt.py:3370-3371`) *after* the warmup block closes.

## Out of scope / open questions

- `alpha < 0.1` not tested: the trend from `alpha = 0.1 → 0.3` suggests negligible effect at lower blend weights, but this was not measured.
- Orders greater than 12 not tested: longer contexts could catch different co-occurrence structure; compute scaling of the C++ hash-table path at higher orders was not characterized in this package.
- Logistic-domain (log-odds) blend alternatives to the probability-domain blend here were not tested.
- Non-TTT eval pipelines were not tested; the negative result is conditional on the phased-LoRA-TTT stack.

## Single-seed scope

This package reports a faithful seed-0 reproduction plus eval-only ablations; it is a non-record evidence package and not a leaderboard claim. This submission uses seed 0 only, both for the reproduction and for the three corrector ablations. The reproduction is compared against #1610's published seed-0 number (1.07216564), not against their 3-seed mean. Multi-seed validation was descoped: with a +1.9×10⁻⁵ BPB delta against the matched seed and a monotonic +0.002 to +0.017 degradation across the corrector grid, additional seeds would refine the variance estimate but are unlikely to flip either direction. The negative-result claim is therefore bounded to seed 0 of the reproduced #1610 checkpoint.

## Artifacts and reproducibility

| File | What it is |
|---|---|
| `train_gpt.py` | This PR's training script; pinned to #1610 upstream with the eval-only-quantized guards applied. |
| `train_seed0.log` | Raw training log for the seed-0 reproduction (script-level timing + per-step metrics; training script writes compact output by design). |
| `ablation_1a.log`, `ablation_1b.log`, `ablation_1c.log` | Raw eval logs for the three corrector configurations (same logging convention). |
| `reproduction_summary.json` | Machine-readable reproduction metrics. |
| `ablation_summary.json` | Machine-readable corrector ablation results (all three configs). |
| `submission.json` | Non-record submission metadata. |
| `requirements.txt` | Python dependencies; pins `torch==2.9.1+cu128`; FA3 notes inline. |
| `provenance/commit_sha.txt` | This branch's commit SHA. |
| `provenance/env_fingerprint.txt` | Torch / CUDA / Python versions at run time. |
| `provenance/hardware_info.txt` | `nvidia-smi` output captured at Gate A. |

**Commit SHA.** `1765afc7d62ce03a1219ca81cc92eea4fabdf343` (pins PR #1610 at upstream `ca1919539dc6e328ea890cb03ad3ca1c5a84da55`; plus eval-only-quantized guards at `e99f18e`).

**Hardware.** 8× NVIDIA H100 80GB HBM3, SXM5, CUDA 12.8, driver 570.211.01.

Supplementary external artifact archive for reproducibility: <https://huggingface.co/amay01/parameter-golf-pr1610-reproduction-artifacts>. Contains the preserved full run tarball (141 MB, MD5 `caf8adf63d8c80965f6671beba95d7aa`): pre-quantization checkpoint, quantized checkpoint, full ablation intermediate artifacts. Not required to reproduce the headline number — `train_gpt.py` and the logs in this folder are self-sufficient.
