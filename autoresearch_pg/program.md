# Mission

You are operating a specialized research harness for the OpenAI Parameter Golf
challenge in this repository.

Your objective is not to improve raw pre-quant validation loss. Your objective
is to improve the final deployed score:

- minimize `final_int8_zlib_roundtrip_exact val_bpb`
- keep `bytes_total <= 16_000_000`
- keep experiments reproducible and promotable to the official `records/` format

# Operating Rules

- Edit only the candidate-local files in `autoresearch_pg/candidates/<candidate_id>/`.
- Do not edit the root `train_gpt.py` directly.
- Prefer one clear hypothesis per candidate.
- Keep changes small enough that failures are attributable.
- Record the intended hypothesis in `notes.md`.
- Prefer curated templates for higher-level ideas when they exist, instead of
  recreating the same concept ad hoc.

# Current Search Priorities

1. Reduce the post-quant gap. The baseline loses a meaningful amount of score at export.
2. Improve data/sample efficiency under the 600s cap.
3. Explore challenge-native ideas:
   - quantization-aware training
   - multi-token prediction heads during training
   - shared-depth / recurrent transformer variants
   - factorized embeddings and tokenizer-aware variants
4. Use and extend the template library for recurring idea families before
   inventing one-off mutations.

# Things To Watch

- Artifact bytes are a hard constraint, not an afterthought.
- Large code changes are fine only if they buy real score.
- A lower pre-quant score that exports badly is not a win.
- Avoid mixing tokenizer changes with architecture changes in the same early candidate.

# Promotion Philosophy

- `smoke_local` should catch broken code and obvious regressions.
- `proxy_1gpu_fast` should filter weak ideas cheaply.
- `proxy_1gpu_full` should be the main local bar before using `8xH100`.
- `track_8gpu_600s` is reserved for candidates that are already compelling.

# Candidate Note Template

Use this structure in candidate `notes.md`:

1. Hypothesis
2. Expected upside
3. Expected risk
4. Exact knobs changed
5. Promotion bar
