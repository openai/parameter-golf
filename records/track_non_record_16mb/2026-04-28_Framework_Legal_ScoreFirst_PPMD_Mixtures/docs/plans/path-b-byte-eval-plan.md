## Plan: Path B Byte Eval

Implement a standalone non-record evaluator for Path B: a normalized byte-level neural predictor computed by marginalizing token probabilities over SP8192 token-byte trie prefixes, then mixing with a proper score-before-update PPM-D byte model. The implementation will prioritize correctness with reference tests, then add GPU-vectorized marginal extraction and a RunPod-ready execution prompt without launching RunPod.

**Phases**
1. **Phase 1: Reference Path B Primitives**
    - **Objective:** Build and test tokenizer byte-sequence/trie primitives plus a CPU reference neural byte marginalizer that proves Path B normalization on synthetic and SP8192-like cases.
    - **Files/Functions to Modify/Create:** `scripts/eval_path_b_ppmd.py`; `tests/test_path_b_ppmd_eval.py`; functions for token byte sequence construction, trie construction, reference marginalization, and byte-stream expansion.
    - **Tests to Write:** `test_reference_marginal_matches_bruteforce`; `test_neural_byte_distribution_normalizes`; `test_terminal_prefix_mass_is_excluded`; `test_sentencepiece_leading_space_modes`; `test_zero_byte_special_tokens_do_not_break_root_distribution`.
    - **Steps:**
        1. Write failing tests for trie byte semantics and neural byte normalization.
        2. Implement minimal reference trie and marginalizer code.
        3. Run the new unit tests and fix correctness issues until they pass.
2. **Phase 2: Optimized Scoring Kernels**
    - **Objective:** Add GPU-friendly interval/cumsum marginalization and proper PPM-D with exclusion, validating them against the reference implementation.
    - **Files/Functions to Modify/Create:** `scripts/eval_path_b_ppmd.py`; `tests/test_path_b_ppmd_eval.py`; optimized trie interval tables, torch cumsum scoring path, PPM-D exclusion state, mixture scoring helpers.
    - **Tests to Write:** `test_optimized_marginal_matches_reference`; `test_ppmd_distribution_normalizes`; `test_mixture_distribution_normalizes`; `test_score_before_update_ordering`.
    - **Steps:**
        1. Write failing parity and normalization tests for optimized neural and PPM-D components.
        2. Implement vectorized torch interval scoring and proper PPM-D exclusion.
        3. Run focused tests and tune numerical tolerances without weakening normalization guarantees.
3. **Phase 3: Fresh Eval CLI Integration**
    - **Objective:** Provide a standalone CLI that loads `results/exp_1876_ppmd/train_gpt_merged.py`, deserializes `prod_8gpu_s42v2/final_model.int6.ptz`, computes Path B records for prefix/full validation, and emits audit JSON without mutating production training code.
    - **Files/Functions to Modify/Create:** `scripts/eval_path_b_ppmd.py`; `tests/test_path_b_ppmd_eval.py`; CLI functions for importing exp_1876 code, loading artifacts, rank-aware shard writing/merging, and JSON output.
    - **Tests to Write:** `test_cli_dry_run_artifact_paths`; `test_shard_merge_preserves_order`; `test_subset_byte_denominator_regression`; `test_eval_output_schema`.
    - **Steps:**
        1. Write failing tests for CLI path validation, shard merge order, denominator metadata, and output schema.
        2. Implement the dry-run/load-path and merge/schema logic first, then wire model-eval entry points behind explicit CLI flags.
        3. Run focused tests and keep full RunPod/GPU execution out of this phase.
4. **Phase 4: Red-Team Review and RunPod Prompt**
    - **Objective:** Document computational cost, red-team risks, mitigations, and a Copilot CLI prompt/execution plan for a safe RunPod non-record eval without running it.
    - **Files/Functions to Modify/Create:** `plans/path-b-byte-eval-redteam.md`; `plans/path-b-byte-eval-runpod-prompt.md`; `plans/path-b-byte-eval-complete.md`.
    - **Tests to Write:** `test_runpod_prompt_does_not_include_secret_or_launch_command`; `test_redteam_doc_mentions_boundary_and_normalization_risks`.
    - **Steps:**
        1. Write tests that enforce the RunPod prompt is non-secret-bearing and clearly instructs not to launch until approved.
        2. Draft the red-team and execution prompt documents with cost estimates and safety gates.
        3. Run the relevant tests and complete the final report.

**Open Questions**
1. Should the first real RunPod eval use TTT for comparability, or sliding-only for a cheaper sanity result first?
2. Should a later productionized version add a compiled PPM-D extension if Python PPM-D is too slow?
3. Should record reviewers require marginalization over all possible token segmentations rather than official token-boundary prefixes?