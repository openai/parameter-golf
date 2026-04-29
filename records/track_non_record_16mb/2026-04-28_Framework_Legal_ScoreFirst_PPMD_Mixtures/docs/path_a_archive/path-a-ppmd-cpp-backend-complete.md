## Plan Complete: Path A PPM-D C++ Backend

Built a pybind11 C++17 backend for the Path A PPM-D evaluator (`scripts/eval_path_a_ppmd.py`) that delivers bit-exact end-to-end BPB equivalence with the Python reference (0.0 absolute diff on 64-position synthetic) and ~17–50× single-thread speedup, with deterministic OpenMP vocab-shard parallelism. All 6 phases used strict TDD (tests-first, run-red, minimal-code, run-green) under `.venv-smoke` (Python 3.12 / pybind11 3.0.4); legacy tests still pass under `/bin/python3.8`. Phase 6 SLURM benchmark (32 cpu_short threads, 4096×8192 synthetic) measured ~2.28M probes/s → full non-record eval projected at ~38 days, ~258× over the 3.5 h short-queue budget — CPU-only gate fails. Follow-on CUDA plan opened in `plans/path-a-ppmd-cuda-backend-plan.md`.

**Phases Completed:** 6 of 6
1. ✅ Phase 1: Build infrastructure + venv pybind11 (Makefile, src skeleton, smoke import)
2. ✅ Phase 2: PPM-D byte-prob kernel + 5 equivalence tests (≥ 15 decimals)
3. ✅ Phase 3: Trie scorer + OpenMP vocab sharding (bit-exact end-to-end BPB; thread-count invariant)
4. ✅ Phase 4: SLURM CPU benchmark scripts (cpu_short routing via --time=03:59:00)
5. ✅ Phase 5: `--backend cpp` integration + real-slice SLURM job (default Python contract preserved)
6. ✅ Phase 6: CUDA decision gate → FAIL → CUDA plan opened (no CUDA implementation)

**All Files Created/Modified:**
- scripts/ppmd_cpp/Makefile
- scripts/ppmd_cpp/src/module.cpp
- scripts/ppmd_cpp/src/sha256.{hpp,cpp}
- scripts/ppmd_cpp/src/ppmd.{hpp,cpp}
- scripts/ppmd_cpp/src/virtual_ppmd.{hpp,cpp}
- scripts/ppmd_cpp/src/backoff.hpp
- scripts/ppmd_cpp/src/trie.{hpp,cpp}
- scripts/ppmd_cpp/src/scorer.{hpp,cpp}
- scripts/ppmd_cpp/bench_cpu.py
- scripts/ppmd_cpp/bench_cpu.sbatch
- scripts/ppmd_cpp/eval_real_slice.sbatch
- scripts/eval_path_a_ppmd.py (extended; default Python path unchanged)
- tests/test_ppmd_cpp_smoke.py
- tests/test_ppmd_cpp_kernel.py
- tests/test_ppmd_cpp_scorer.py
- tests/test_ppmd_cpp_bench_cli.py
- tests/test_ppmd_cpp_backend_dispatch.py
- plans/path-a-ppmd-cpp-backend-plan.md
- plans/path-a-ppmd-cpp-backend-phase-{1,2,3,4,5,6}-complete.md
- plans/path-a-ppmd-cuda-backend-plan.md

**Key Functions/Classes Added:**
- C++ side: `PPMDState`, `VirtualPPMDState`, `Trie`, `pack_ctx`, `dfs_subtree`, `score_path_a_arrays`, `combine_path_a_partials`, `trie_partial_z_and_target`, vendored SHA-256
- Python side: `_pack_vocab_for_cpp`, `_import_ppmd_cpp`, `_score_path_a_arrays_cpp`, `_score_path_a_arrays_dispatch`
- CLI: `--backend {python,cpp}`, `--backend-equiv-check K`, `--positions`, `--results-name`

**Test Coverage:**
- Total cpp-stack tests written: 18 (smoke 2 + kernel 5 + scorer 5 + bench-CLI 3 + backend-dispatch 3)
- Legacy Python-only tests still passing: 11
- All tests passing: ✅
- End-to-end BPB equivalence (Python ↔ C++): 0.0 absolute diff on 64-position synthetic
- OpenMP determinism: identical BPB across 1/2/4/8 threads

**Performance Summary:**
- Python reference (single-thread): baseline
- C++ single-thread synthetic: ~17–50× speedup
- C++ 32-thread cpu_short SLURM (4096×8192 synthetic): 2.28M probes/s
- Projected full non-record eval (CPU-only): ~38 days → 258× over short-queue budget → CUDA required

**Recommendations for Next Steps:**
- Execute `plans/path-a-ppmd-cuda-backend-plan.md` (5-phase CUDA backend, A100-first via `--gres=gpu:a100:1`)
- Address Phase 5 minor non-blocking recommendations: (a) soft-fallback should record `backend="python"` in report when extension import fails; (b) tighten `test_backend_flag_defaults_to_python` to assert exact equality not substring; (c) emit runtime warning when `--backend cpp` silently drops `max_positions`/`normalization_sample_every`
- Remove stale comment at scorer.cpp:167 referencing nonexistent `_build_py_candidates`
- Once CUDA backend lands, retire the `_score_path_a_arrays_cpp` soft-fallback path or convert to a hard-fail gate
