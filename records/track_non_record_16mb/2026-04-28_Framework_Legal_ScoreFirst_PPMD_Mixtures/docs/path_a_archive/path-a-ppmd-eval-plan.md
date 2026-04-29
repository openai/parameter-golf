## Plan: Path A PPM-D Eval

Build a standalone, auditable Path A evaluator for exp_1876 that computes a token-normalized PPM-D mixture over all SP8192 token IDs. The first implementation prioritizes mathematical correctness, deterministic audit outputs, and RunPod-ready orchestration, while exposing prefix limits and rank-sharded candidate normalization for performance experiments on 8×H100.

**Phases**
1. **Phase 1: Evaluator Architecture**
    - **Objective:** Define and scaffold a standalone eval-only script that can import the exp_1876 model code without running training.
    - **Files/Functions to Modify/Create:**
        - Create `scripts/eval_path_a_ppmd.py`
        - Create `tests/test_path_a_ppmd_eval.py`
    - **Tests to Write:**
        - `test_candidate_token_bytes_include_sentencepiece_space`
        - `test_path_a_mixture_normalizes_over_tokens`
    - **Steps:**
        1. Write tests for candidate byte construction and token-normalized mixture math.
        2. Implement script scaffolding with exact PPM-D state, candidate byte handling, and Path A scoring core.
        3. Run focused unit tests.

2. **Phase 2: Eval Integration Hooks**
    - **Objective:** Add CLI hooks for loading exp_1876 artifacts, collecting neural per-position NLLs, and running prefix/full Path A scoring.
    - **Files/Functions to Modify/Create:**
        - Extend `scripts/eval_path_a_ppmd.py`
        - Extend `tests/test_path_a_ppmd_eval.py`
    - **Tests to Write:**
        - `test_score_first_state_hash_changes_only_after_update`
        - `test_rank_sharded_vocab_reduction_matches_single_rank`
    - **Steps:**
        1. Write tests for score-first state behavior and candidate-shard reduction.
        2. Implement CLI modes, artifact paths, audit JSON writing, and distributed-safe reduction math.
        3. Run focused unit tests.

3. **Phase 3: Red-Team and RunPod Plan**
    - **Objective:** Red-team correctness/performance hazards and create a non-executing RunPod plan plus Copilot CLI prompt.
    - **Files/Functions to Modify/Create:**
        - Create `plans/path-a-ppmd-runpod-plan.md`
        - Create `plans/path-a-ppmd-eval-complete.md`
    - **Tests to Write:**
        - `test_cli_help_runs_without_exp_import`
        - `test_audit_schema_for_prefix_eval`
    - **Steps:**
        1. Add non-GPU CLI smoke and audit schema tests.
        2. Draft red-team section covering normalization, state causality, distributed synchronization, and cost risk.
        3. Write the RunPod execution plan and prompt without launching any pod.

**Open Questions**
1. Should the first RunPod smoke evaluate sliding-only neural NLLs or full TTT scoring?
2. Should full Path A initially run on rank 0 for proof, or rank-sharded candidate normalization for speed?
3. What maximum non-record RunPod budget should gate a full Path A eval attempt?
