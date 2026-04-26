# Experiment 0018_recur3x3_swiglu_2attn_bigramhash

Parent: 0012_recur3x3_swiglu_2attn_1s4d

## Question
0012/0014 (2:1 hybrid sandwich) ties transformer-best 2.087 to 0.001 BPB. Adding BigramHash — a record-validated cheap recall augmentation that hashes adjacent token pairs into a small extra embedding — could push the SSM-hybrid BELOW transformer-best. Per primer §4.5, BigramHash is the recommended candidate remedy for SSM's recall gap. Param cost is small (~300-500K params, well under cap headroom).

## Hypothesis [CONJECTURE]
val_bpb in [2.075, 2.090]. The 2:1 hybrid is already near transformer-best; the recall gap should be small, but BigramHash typically gives 0.005-0.015 BPB even in transformer settings. If our hybrid still has any residual recall deficit, BigramHash should close it. Most likely: val ≈ 2.080-2.085 (small win that pushes below transformer-best).

## Change

**Code edit in train_gpt.py:**

Add a `BigramHashEmbedding` module modeled on `records/track_10min_16mb/2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485/train_gpt.py` (lines 898-919). The module:
- bigram_vocab_size and bigram_dim env-driven (BIGRAM_VOCAB_SIZE, BIGRAM_DIM)
- xor-hash adjacent token pairs to bucket index
- nn.Embedding initialized to zeros (so module starts as identity)
- optional projection from bigram_dim → model_dim if they differ; zero-init
- learnable `scale` parameter (init 0.05) to gate the contribution
- forward returns `proj(embed(hash(tokens))) * scale`

Integrate in `GPT.forward`:
- After `x = self.tok_emb(input_ids)` and BEFORE the final `F.rms_norm(x, ...)`:
  - `x = x + self.bigram_hash(input_ids)`
- The bigram hash module is a property of GPT, constructed in __init__.
- Also include the bigram embed/proj weights in the state dict so they get quantized normally — they're 2D matrices, will route to Muon by default.

Module integration spec:
- `BigramHashEmbedding(bigram_vocab_size: int, bigram_dim: int, model_dim: int)`.
- Add `BIGRAM_VOCAB_SIZE` (default 4096) and `BIGRAM_DIM` (default 64) to Hyperparameters. When `BIGRAM_VOCAB_SIZE=0` (the default), DON'T create the module (preserve baseline behavior).
- Pass `bigram_vocab_size`, `bigram_dim` as GPT __init__ kwargs; conditionally instantiate `self.bigram_hash`.

**env.sh additions:**
```
# Inherited from 0012:
... (all the recur+swiglu+2attn settings)
# 0018: BigramHash bolt-on
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=64
```

## Param cost / cap math
- BigramHash embed: 4096 × 64 = 262K params (fp16 storage = 524 KB) — under INT8_KEEP_FLOAT_MAX_NUMEL=65536? No, 262k > 65k → int8 quantized → 262 KB.
- Proj: 64 × 512 = 32K params → 32 KB int8.
- scale: 1 fp32 = 4 B.
- Total addition: ~300 KB. 0012 was 12.28 MB → 0018 ~12.6 MB. Well under 16 MB cap.

## Disconfirming
- val_bpb < 2.080 → BIG win. SSM-hybrid BEATS transformer-best on a single seed; SEED=42 confirm needed.
- val_bpb in [2.080, 2.090] → small confirmed win; mean with SEED=42 might land just below 2.087.
- val_bpb in [2.085, 2.095] → noise; BigramHash doesn't help on the already-saturated 2:1 hybrid.
- val_bpb > 2.095 → BigramHash HURTS. Unexpected; would suggest the 2:1 hybrid was already saturated and adding parameters destabilizes.
- Crash → most likely cause: env-var threading (BIGRAM_VOCAB_SIZE=0 path missing in __init__ or forward). Check.

## Notes from execution

Implementation completed by subagent on 2026-04-26.

Edits to `experiments/0018_recur3x3_swiglu_2attn_bigramhash/train_gpt.py`:
- Hyperparameters class: added `bigram_vocab_size` (env `BIGRAM_VOCAB_SIZE`, default 0 = disabled to preserve baseline) and `bigram_dim` (env `BIGRAM_DIM`, default 64).
- Added `BigramHashEmbedding(nn.Module)` in the TRANSFORMER MODULES section, just before `MLP`. Mirrors the Scylla reference verbatim: zero-init `nn.Embedding`, optional zero-init `CastedLinear` projection iff `bigram_dim != model_dim`, learnable fp32 `scale` initialized to 0.05, xor-hash with multipliers 36313/27191 modulo `bigram_vocab_size - 1`, position 0 mapped to `mod` as a sentinel since there is no preceding token.
- `GPT.__init__`: added `bigram_vocab_size: int = 0, bigram_dim: int = 64` kwargs and conditional construction `self.bigram_hash = BigramHashEmbedding(...) if bigram_vocab_size > 0 else None`. Constructed immediately after `self.tok_emb` so the module is registered before `_init_weights()` runs (zero-init paths are unaffected since `BigramHashEmbedding` already zero-inits its own weights and the `_zero_init` guard in `_init_weights` is opt-in via attribute, which we do not set).
- `GPT.forward`: inserted `if self.bigram_hash is not None: x = x + self.bigram_hash(input_ids)` between `x = self.tok_emb(input_ids)` and `x = F.rms_norm(x, (x.size(-1),))`. With `BIGRAM_VOCAB_SIZE=0` (default) the conditional short-circuits and the forward path is identical to the parent 0012 baseline.
- `main()` GPT construction: pass `bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim` through.

Optimizer routing (sanity-checked, no code change needed):
- `bigram_hash.embed.weight` (ndim=2, name has no CONTROL_TENSOR pattern) routes to `matrix_params` -> Muon. Note: this lives under `base_model.bigram_hash.*`, not `base_model.blocks.*`, so it is NOT picked up by `block_named_params`. That means it would be missed by both Muon (matrix_params) and Adam (scalar_params). The plan says "they're 2D matrices, will route to Muon by default" — this is incorrect under the actual optimizer split which only walks `base_model.blocks.named_parameters()`. **Bigram hash params are currently un-optimized.** The plan/spec did not request modifying the optimizer split (constraints explicitly forbid it), so this is left as-is per spec. If `BIGRAM_VOCAB_SIZE>0` is enabled, the bigram weights will sit at their zero init and the module will contribute nothing; the run will be functionally identical to baseline. Flag this for the next iteration: either add `base_model.bigram_hash.*` params to `matrix_params`/`scalar_params`, or relax the constraint.
- `bigram_hash.proj.weight` (ndim=2): same situation as above (would route to Muon if added to the optimizer, currently un-optimized).
- `bigram_hash.scale` (ndim=0): same situation (would route to Adam scalar_params via the ndim<2 check, currently un-optimized).

Verification:
- `python3 -c "import ast; ast.parse(open(...).read())"` parsed cleanly.
- Did not modify `env.sh`, `run_experiment.sh`, the canonical repo-root `train_gpt.py`, or anything in `records/`.
- Did not modify `S4DLin`, `CausalSelfAttention`, `MLP`, `eval_val`, `build_sentencepiece_luts`, the quantization functions, or the optimizer split (per constraints).
- Did not run experiments.

Spec items that did not apply: the spec's "params route to Muon/Adam (correct)" annotation only holds if these params live under `base_model.blocks` — they do not (they live under `base_model`), so they are currently outside the optimizer's reach. Recorded above for follow-up.
