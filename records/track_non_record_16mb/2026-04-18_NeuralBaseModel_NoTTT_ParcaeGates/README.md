# Non-record submission — Neural Base Model, No TTT

**val_bpb = 1.07706** (sliding-window eval, seed 1337, non-casefold SP8192) | **15,962,729 B (~15.96 MB)** | 8×H100 80GB SXM | 596s training + sliding-window eval

## Summary

- **No test-time training.** No LoRA adapters, no global-SGD. Pure architectural + quantization result.
- **Beats current merged non-casefold SOTA ([PR #1493](https://github.com/openai/parameter-golf/pull/1493) @bigbag, 1.0810)** by **−0.00394 BPB (−0.00273 nats)** — without any test-time adaptation.
- **Non-record submission.** Doesn't attempt to clear the 0.005-nat bar vs merged SOTA (that's the full-TTT path, for which I'm out of compute credits). Positioning: establishes the *base-model ceiling* for standard SP8192 architectures and gives future TTT work a cleaner lift measurement.

## Positioning

Most competitive submissions on the main-track leaderboard include test-time training, which conflates *architectural* gains with *test-time compute* gains. This submission isolates the architectural contribution:

- val_bpb = 1.07706 with `TTT_ENABLED=0` in the shipped config
- TTT scaffolding (phased global-SGD + per-doc LoRA, ported from PR #1693 @dexhunter + @MarioPaerle) remains in the file for experiments but is disabled by default
- Running with `TTT_ENABLED=1` on the same model trends toward 1.074–1.075 in my experiments, short of clearing the 0.005-nat bar vs the casefold-track PR #1695 (1.0759) — further TTT tuning is future work pending compute

As a reference point: PR #1493 (merged SOTA) reports sliding-only 1.0829 and post-TTT 1.0810. Our sliding-only is 1.07706, **below their TTT-enabled number**, and below their sliding-only by **0.00584 BPB**.

## Results (single seed)

| Stage | val_bpb | Notes |
|---|---|---|
| Pre-quant EMA (bf16) | **1.0699** | End-of-training, post-EMA |
| Post-quant (int6+int7+brotli) sliding | **1.07706** | **Submission number** |
| Quantization cost | +0.00716 | Typical for int6 GPTQ |

- Training: **596s** (wallclock cap), step 4602/20000
- Artifact: **15,962,729 B** (under 16,000,000 B cap by 37,271 B)
- Seed: 1337 (single-seed result — compute-constrained, see Note on logs and seeds below)

## Architecture

Inherited from [PR #1674](https://github.com/openai/parameter-golf/pull/1674) (ours, non-record research submission):

- **Parcae Constrained Loop Injection** — SSM-style boundary condition at each loop re-entry: `x = A_bar * x + B_bar * x0` with learned per-dim `loop_log_A` / `loop_delta` / `loop_B`. `A_bar ∈ (0, 1)` by construction (softplus on delta, exp of negative exp on log_A) enforces bounded-decay; `B_bar` re-injects the original residual stream. Three per-dim scalars total.
- **Gemma-style Global / Local Attention** — `global_attn_layers=[4, 9, 10]` get full causal attention + partial RoPE (`rope_dims=16 / head_dim=64`); remaining layers use sliding-window attention + full RoPE for positional precision within the window.
- **Gram Newton-Schulz** for high-aspect-ratio MLP banks (α > 2.5) — reduces Newton-Schulz cost on `mlp_up_bank` (4:1 ratio) and `mlp_down_bank`. NS steps dropped 5 → 4 since the architecture no longer requires the extra refine step.

Inherited from [PR #1530](https://github.com/openai/parameter-golf/pull/1530) (@samacqua):

- **Variable-length attention** — `flash_attn_varlen_func` with `cu_seqlens` boundaries; training, eval, and global-SGD TTT (when enabled) never attend across unrelated documents packed in the same flat batch.
- **Fused MLP triton kernel** — custom `linear_xielu_kernel` fuses the up-projection + xIELU activation + squaring into a single kernel (analogue of @samacqua's `linear_leaky_relu_square_kernel` with our xIELU activation).

Inherited from [PR #1693](https://github.com/openai/parameter-golf/pull/1693) (@dexhunter + @MarioPaerle) — **gates used, TTT disabled for this submission**:

- **Attention Output Gate** ([PR #1667](https://github.com/openai/parameter-golf/pull/1667) @MarioPaerle) — per-head input-dependent `sigmoid × 2` gate on attention output, zero-init for identity-at-init, composed with `fullgraph=True` compile.
- **SmearGate** (@kellerjordan concept via modded-nanogpt; @MarioPaerle reintroduction) — input-dependent per-channel residual mixer blending current token with previous token (strictly causal, backward-looking by one position), zero-init lambda.

**New in this PR:**

- **Layered Local Sliding Windows** — a prior uniform-window ablation on this architecture showed `LOCAL_WINDOW_SIZE=512` and `LOCAL_WINDOW_SIZE=1024` produced identical val_bpb, suggesting per-layer window size is a near-free dial. This PR splits: 512 tokens on locals `{0, 1, 2, 3, 5}` (early layers + the recurrence-loop tail at layer 5, where attention FLOPs are 2×-amplified by `num_loops=2`), 1024 tokens on locals `{6, 7, 8}` (post-loop integration layers where wider context plausibly helps and isn't loop-amplified). Global layers `{4, 9, 10}` retain full attention. Zero compile-cost — each block's `attn.window_size` is set once at init and baked as a per-subgraph constant.

Dropped from PR #1674:

- **KV-tying on global attention layers** — present in PR #1674 as a memory/param-budget optimization, disabled here (`KV_TIE_GLOBAL=0`). Freed V-weights are spent on more expressive global-layer attention rather than on looser quantization clipping.

## Quantization

- **int6 GPTQ** on matrix weights (Q/K/V/O banks, MLP banks) with SDClip (std-based clipping, k=12.85 for matrix), 16 calibration batches, 4s reserved from training budget
- **int7 GPTQ** on embedding (`EMBED_BITS=7`, clip k=15.0)
- **Brotli** on the quantized state dict
- **LZMA** on the code

Total artifact at seed 1337: **15,962,729 B** (compressed code + quantized state). Under the 16 MB cap by 37 KB.

## Compliance (Issue #1017 Track B)

Since this submission runs the sliding-window eval path with no test-time adaptation, only the causality and normalization conditions apply:

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. `flash_attn_3_func(..., causal=True, window_size=attn.window_size)` on every attention call. SmearGate mixes with the *previous* token only (`F.pad(x[:, :-1], (0, 0, 1, 0))`).
- **Condition 2 (Normalized distribution):** Standard softmax over full SP8192 vocabulary. Gates modulate hidden states, not logits. `logit_softcap * tanh(logits / logit_softcap)` applied uniformly (standard stabilization, not a selective modulation).

Conditions 3 and 4 (score-before-update, single-pass) are TTT-specific and don't apply here.

**Tokenizer:** standard SP8192 (Kevin Clark's pre-tokenized dataset via [PR #78](https://github.com/openai/parameter-golf/pull/78) @mtybadger). No casefold — legality-independent of Issue #1604.

## Note on logs and seeds

**No training/eval log attached.** The VM used for these runs went down before the seed-1337 log could be pushed to GitHub, and I no longer have GPU access to reproduce. The metrics (val_bpb 1.07706, artifact 15,962,729 B, train 596s) are recorded from my own observation of the run output during the session. I invite judges to reproduce using the command below and expect the numbers to be within normal seed-variance.

**Single-seed result (seed 1337).** Compute-constrained, consistent with the "non-record research submission" convention used by [PR #1674](https://github.com/openai/parameter-golf/pull/1674) (ours, earlier). Seed was picked before the run, not hindsight-selected.

## Reproduction

```bash
# setup
uv sync                      # torch 2.11 cu130 + flash-attn-3 from pyproject.toml
nvidia-smi | head -20        # confirm 8x H100 80GB SXM

# run (all defaults — TTT off, sliding on, non-casefold SP8192)
ARTIFACT_DIR=runs/base_model_seed1337 SEED=1337 \
  uv run torchrun --standalone --nproc_per_node=8 --max-restarts=0 \
  train_gpt.py \
  > runs/base_model_seed1337/run.log 2>&1
```

Expected log markers:
- `Total submission size quantized+brotli: ~15,962,729 bytes`
- `diagnostic quantized_sliding_window val_loss:... val_bpb:1.07706...`
- Total wallclock: ~700-800s (596s training + ~100-200s eval including quantization)

Running with `TTT_ENABLED=1` additionally invokes the phased TTT path (ported from PR #1693), but this is *not* the submission metric.

## Lineage

- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) (@samacqua, varlen attention + fused MLP + doc-independent LoRA TTT) →
- [PR #1586](https://github.com/openai/parameter-golf/pull/1586) / [PR #1648](https://github.com/openai/parameter-golf/pull/1648) (xIELU + QK-Gain) →
- [PR #1674](https://github.com/openai/parameter-golf/pull/1674) (ours: Parcae + Gemma-style attn + Gram NS + KV-tying, non-record) →
- **this PR** (+ AttnOutGate + SmearGate + layered local windows, KV-tying dropped, no TTT)

Parallel track (with TTT):
- [PR #1693](https://github.com/openai/parameter-golf/pull/1693) (@dexhunter + @MarioPaerle, casefold + gates + phased TTT) →
- [PR #1695](https://github.com/openai/parameter-golf/pull/1695) (@X-Abhishek-X, non-casefold + gates + phased TTT + LoRA+ / layer-LR-alpha)

## Credits

- **@samacqua** — [PR #1530](https://github.com/openai/parameter-golf/pull/1530) base (varlen attention, fused MLP, document-independent TTT framework, compressed-artifact tooling)
- **@bigbag** — [PR #1493](https://github.com/openai/parameter-golf/pull/1493) merged non-casefold SOTA (comparison baseline)
- **@clarkkev** — [PR #1394](https://github.com/openai/parameter-golf/pull/1394) GPTQ SDClip pipeline and SP8192 tokenizer integration
- **@MarioPaerle** — [PR #1667](https://github.com/openai/parameter-golf/pull/1667) Attention Output Gate; SmearGate reintroduction to parameter-golf
- **@kellerjordan** — SmearGate concept (modded-nanogpt)
- **@dexhunter** — Multi-Phase Global SGD TTT framework (PR #1610 / #1626 / #1670 / #1693) — present in the file (disabled here) for future TTT work
- **@mtybadger** — [PR #78](https://github.com/openai/parameter-golf/pull/78) SP8192 tokenizer + pre-tokenized dataset
- **@mikeapedia** — [PR #1674](https://github.com/openai/parameter-golf/pull/1674) base carried into this submission (Parcae loop, Gemma-style attention, Gram Newton-Schulz, xIELU activation); new here: layered local sliding windows, non-KV-tied global attention, file defaults for non-TTT submission

## Test plan

- [x] Single-seed training on 8×H100 SXM (seed 1337) — 596s train, under 600s cap
- [x] Artifact size 15,962,729 B — under 16,000,000 B cap
- [x] Sliding-window eval completes with val_bpb 1.07706
- [x] `TTT_ENABLED=0` is the shipped default — submission reproduces the 1.07706 number without TTT
- [x] Standard SP8192 tokenizer, Track B conditions 1-2 satisfied (causality, normalized distribution)
- [x] Tested `EVAL_EXTRA_LOOPS` (extra recurrence iterations at eval time) — no improvement, regresses sliding bpb. Submission ships with default `EVAL_EXTRA_LOOPS=0`.
- [ ] Judges verify reproducibility on their infrastructure (training/eval log not attached — see note above)
