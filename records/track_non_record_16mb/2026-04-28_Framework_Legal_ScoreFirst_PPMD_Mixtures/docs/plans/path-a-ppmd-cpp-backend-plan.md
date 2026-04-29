## Plan: Path A PPM-D C++ Backend

Build a compiled C++ reference backend for the Path A token-normalized PPM-D evaluator (currently in [scripts/eval_path_a_ppmd.py](scripts/eval_path_a_ppmd.py)) so full non-record validation finishes inside one SLURM short-queue allocation (`<= 03:59:00`), and to lay the groundwork for an optional CUDA backend if record-track speed becomes necessary. The Python reference impl is the conformance contract: byte-probability values must match `_ppmd_byte_prob_with_provider` to `>= 15` decimal places and shard reductions to `>= 14`.

### Environment & constraint summary (from research)

- Workstation has **no usable GPU** (`nvidia-smi` fails). All GPU work routes through SLURM.
- System `/bin/python3.8` has **no `Python.h`** and `python3.8-config` is missing. Use the existing `.venv-smoke` (Python 3.12, headers present) and `uv pip install pybind11` into it.
- No `cmake` on `PATH`. Use a hand-written Makefile invoking `g++ 8.5.0` (or a newer compiler from a `medsci` module if needed for OpenMP performance).
- SLURM: do **NOT** specify `--partition`. With `--time=03:59:00` jobs land in `cpu_short`/`gpu_short` automatically. V100 nodes `g001-019` (`gres=gpu:v100:N`), A100 nodes `g020-025` (`gres=gpu:a100:N`).
- Sequential constraint: PPM state advances over positions; the only safe parallelism is **per-position vocab-trie sharding** (already proven exact by `test_rank_sharded_vocab_reduction_matches_single_rank`) plus OpenMP threads within one position.
- Numerical contract: `double` throughout; never `float`. Reproduce backoff order, exclusion semantics, and the all-active branch exactly.

**Phases (6)**

1. **Phase 1: Build infrastructure + venv pybind11**
   - **Objective:** Establish a reproducible build path that produces an importable `_ppmd_cpp` extension module from the workstation login node.
   - **Files/Functions to Modify/Create:**
     - `scripts/ppmd_cpp/` (new directory)
     - `scripts/ppmd_cpp/Makefile` — invokes `g++ -O3 -march=native -fopenmp -fPIC -shared`, derives Python include / extension suffix from `python -c 'import sysconfig; ...'`, links via `pybind11 --includes`.
     - `scripts/ppmd_cpp/src/module.cpp` — minimal pybind11 module exposing `version()` only.
     - `scripts/ppmd_cpp/README.md` — documents `.venv-smoke` activation, `uv pip install pybind11`, `make`, smoke `python -c 'import _ppmd_cpp; print(_ppmd_cpp.version())'`.
     - `tests/test_ppmd_cpp_smoke.py` — skips if `.venv-smoke` not active or extension not built; otherwise asserts import + `version()` returns a string.
   - **Tests to Write:**
     - `test_ppmd_cpp_extension_imports`
     - `test_ppmd_cpp_version_string`
   - **Steps:**
     1. Verify `.venv-smoke/bin/python -c 'import sys; print(sys.version)'` reports 3.12.x.
     2. Install pybind11 into the venv: `.venv-smoke/bin/python -m pip install pybind11`.
     3. Write `module.cpp` with a single `m.def("version", []{ return "0.0.1"; })`.
     4. Write `Makefile` (no cmake), targets `all`, `clean`. Output `scripts/ppmd_cpp/_ppmd_cpp$(EXT_SUFFIX)`.
     5. Write the two unittest tests gated on the extension's presence (skip cleanly if absent).
     6. Run the tests: `.venv-smoke/bin/python -m unittest tests.test_ppmd_cpp_smoke -v`. They must fail until the build succeeds, then pass after `make`.

2. **Phase 2: C++ PPM-D byte-prob kernel + equivalence tests**
   - **Objective:** Port `_ppmd_byte_prob_with_provider` and `_ppmd_byte_probs_with_provider` to C++ with bit-for-bit numerical agreement (`>= 15` decimals).
   - **Files/Functions to Modify/Create:**
     - `scripts/ppmd_cpp/src/ppmd.hpp` — declares `class PPMDState` with packed-context storage (encode 0..5 bytes + length into one `uint64_t`, hashed by `std::unordered_map<uint64_t, std::array<uint32_t,256>>`).
     - `scripts/ppmd_cpp/src/ppmd.cpp` — implements `update_byte(uint8_t b)`, `byte_prob(uint8_t b) -> double`, `byte_probs() -> std::array<double,256>`, `clone_virtual() -> VirtualPPMDState`, `state_digest() -> std::string`.
     - `scripts/ppmd_cpp/src/virtual_ppmd.hpp/.cpp` — `VirtualPPMDState` with overlay map; `fork_and_update` returns a new instance (deep-copy overlay only, share base by pointer).
     - `scripts/ppmd_cpp/src/module.cpp` — extend pybind11 bindings for both classes.
     - `tests/test_ppmd_cpp_kernel.py` — equivalence tests against Python.
   - **Tests to Write:**
     - `test_ppmd_cpp_byte_prob_matches_python_reference` (≥15 decimals over a fuzz set of 200 contexts × all 256 bytes)
     - `test_ppmd_cpp_byte_probs_sums_to_one`
     - `test_ppmd_cpp_state_digest_matches_python` (after a deterministic 1000-byte feed)
     - `test_ppmd_cpp_virtual_fork_and_update_does_not_mutate_base`
     - `test_ppmd_cpp_score_first_digest_invariance`
   - **Steps:**
     1. Write Python-side fuzz harness that builds a `PPMDState` Python and a C++ `PPMDState` from the same seed bytes and asserts `byte_prob(b)` agreement; expect failures.
     2. Implement context packing in `ppmd.hpp` (3 bits length, 5×8 bits = 40 bits payload, fits in `uint64_t`).
     3. Implement `update_byte` mirroring Python: increment all suffix orders for the new byte, then trim window.
     4. Implement `_ppmd_byte_probs_with_provider` directly in C++ with the assigned-set `std::bitset<256>`, escape mass `double`.
     5. Implement single-byte fast path matching `_ppmd_byte_prob_with_provider` early-exit semantics.
     6. Implement `state_digest` using OpenSSL or a vendored SHA-256 (header-only) over the same canonical byte sequence as Python.
     7. Add pybind11 bindings, rebuild, re-run tests.

3. **Phase 3: C++ candidate-trie scorer + vocab sharding**
   - **Objective:** Port `trie_partial_z_and_target`, `combine_path_a_partials`, and `score_path_a_arrays` to C++; expose end-to-end scoring; OpenMP-parallelize the per-position trie DFS by vocab shard.
   - **Files/Functions to Modify/Create:**
     - `scripts/ppmd_cpp/src/trie.hpp/.cpp` — flat-arena trie (`std::vector<int32_t> children_offsets`, `std::vector<uint8_t> children_bytes`, `std::vector<int32_t> terminal_token_ids`, `std::vector<int32_t> terminal_starts`).
     - `scripts/ppmd_cpp/src/scorer.hpp/.cpp` — `trie_partial_z_and_target(virtual_state, trie, shard_start, shard_end) -> {z, target_q, terminal_count}`; `score_path_a_arrays(target_ids, prev_ids, nll_nats, vocab_tables, hyperparams) -> {total_bits, total_bytes, bpb, samples}`.
     - `scripts/ppmd_cpp/src/module.cpp` — pybind11 bindings.
     - `tests/test_ppmd_cpp_scorer.py`.
   - **Tests to Write:**
     - `test_ppmd_cpp_trie_partial_matches_python_single_shard` (≥14 decimals)
     - `test_ppmd_cpp_trie_shard_reduction_exact` (split vocab in two, sum equals single-shard)
     - `test_ppmd_cpp_score_path_a_arrays_matches_python` end-to-end on a 256-position synthetic stream, BPB agreement to ≥10 decimals.
     - `test_ppmd_cpp_score_first_invariant_holds_across_arrays` (state digest before == after score, != after update).
     - `test_ppmd_cpp_openmp_thread_count_does_not_change_result` (run with `OMP_NUM_THREADS=1` and `=4`, BPB identical).
   - **Steps:**
     1. Build trie arena from Python `TrieNode` via a flatten helper exposed back to C++ as numpy arrays.
     2. Implement DFS with explicit stack (avoid recursion blowup on deep tries).
     3. Add OpenMP `#pragma omp parallel for reduction(+:z,target_q,terminal_count) schedule(dynamic)` over root children within `[shard_start, shard_end)`.
     4. Implement `score_path_a_arrays` loop sequentially over positions; each position calls the parallel trie scorer.
     5. Run equivalence tests; iterate until ≥10-decimal BPB match.

4. **Phase 4: SLURM CPU benchmark scripts**
   - **Objective:** Establish per-second probe throughput on `cpu_short`-routed nodes and project full-eval wallclock.
   - **Files/Functions to Modify/Create:**
     - `scripts/ppmd_cpp/bench_cpu.sbatch` — `#SBATCH --time=03:59:00`, `--cpus-per-task=32`, `--mem=64G`, `--ntasks=1`, no `--partition`, no `--gres`.
     - `scripts/ppmd_cpp/bench_cpu.py` — CLI taking `--positions N --vocab-shards K --threads T`, loads a fixed prefix slice, runs `score_path_a_arrays` C++, prints throughput JSON to stdout + a results file under `results/ppmd_cpp_bench/`.
     - `tests/test_ppmd_cpp_bench_cli.py` — sanity test for argument parsing only.
   - **Tests to Write:**
     - `test_bench_cpu_cli_parses`
     - `test_bench_cpu_writes_results_json`
   - **Steps:**
     1. Write `bench_cpu.py` with three modes: `--mode synthetic` (random PPM contexts to measure raw probe rate), `--mode prefix-slice` (real first-N positions of the val stream).
     2. Write `bench_cpu.sbatch` that activates `.venv-smoke`, sets `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, runs the CLI, copies stdout/results to `results/ppmd_cpp_bench/${SLURM_JOB_ID}/`.
     3. CPU-only smoke locally on the login node: `OMP_NUM_THREADS=4 .venv-smoke/bin/python scripts/ppmd_cpp/bench_cpu.py --mode synthetic --positions 256`.
     4. Submit via `sbatch scripts/ppmd_cpp/bench_cpu.sbatch` and confirm the job lands in `cpu_short` (`squeue --me -o '%i %P %T %M %l'`).
     5. Verify results file is written and contains `probes_per_second`, `wallclock_seconds`, `projected_full_eval_seconds`.

5. **Phase 5: Path A integration — `--backend cpp` flag + real-shard SLURM job**
   - **Objective:** Wire the C++ backend into [scripts/eval_path_a_ppmd.py](scripts/eval_path_a_ppmd.py) and validate BPB equivalence on a real prefix slice via SLURM.
   - **Files/Functions to Modify/Create:**
     - `scripts/eval_path_a_ppmd.py` — add `--backend {python,cpp}` flag; when `cpp` selected, route `score_path_a_arrays` to `_ppmd_cpp.score_path_a_arrays(...)`. Default remains `python`.
     - `scripts/ppmd_cpp/eval_real_slice.sbatch` — `--time=03:59:00 --cpus-per-task=32`, runs `eval_path_a_ppmd.py --backend cpp --positions 5000 --backend-equiv-check 64`.
     - `tests/test_ppmd_cpp_backend_dispatch.py` — verifies the dispatch flag selects the right backend and that a 64-position smoke matches Python to ≥10 decimals.
   - **Tests to Write:**
     - `test_backend_flag_defaults_to_python`
     - `test_backend_cpp_dispatches_to_extension_when_available`
     - `test_backend_cpp_bpb_matches_python_64_positions` (≥10 decimals; skip if extension absent)
   - **Steps:**
     1. Add `--backend` flag and a thin dispatch wrapper that imports `_ppmd_cpp` lazily.
     2. Add `--backend-equiv-check K`: after C++ scoring, also score the first K positions with Python and assert BPB agreement; abort otherwise.
     3. Submit a 5,000-position real-slice job; confirm short queue placement; collect results and equivalence diff.
     4. Document the run in `plans/path-a-ppmd-cpp-backend-real-slice-results.md` (NOT a file we create here; only the script writes a JSON in `results/`).

6. **Phase 6: CUDA decision gate**
   - **Objective:** Decide whether to spec a CUDA backend based on measured CPU throughput.
   - **Files/Functions to Modify/Create:**
     - `plans/path-a-ppmd-cpp-backend-complete.md` — final report (written by Conductor, not the implementer).
     - If CUDA needed: `plans/path-a-ppmd-cuda-backend-plan.md` — separate follow-on plan, NOT implemented in this plan.
   - **Tests to Write:** none in this phase.
   - **Steps:**
     1. Take projected full-eval wallclock from Phase 4 + measured BPB-equivalence from Phase 5.
     2. **Pass criterion:** projected wallclock <= 3.5 hours on a single `cpu_short` 32-core node. Stop here.
     3. **Fail criterion:** projected wallclock > 3.5 hours. Open `plans/path-a-ppmd-cuda-backend-plan.md` describing: A100 vs V100 selection (`--gres=gpu:a100:1`), per-position CUDA kernel design (parent-child packed trie, 256 threads/block per-byte parallelism), and the equivalence gate `>= 12` decimals on bf16-disabled `double` math. Do not implement.

**Open Questions**
1. Use `.venv-smoke` (Py 3.12) or `module load anaconda3`? Recommend `.venv-smoke`.
2. Trie storage — flat arena or `vector<TrieNode>`? Recommend flat arena.
3. Context-key encoding — packed `uint64_t` (3-bit length + 40-bit payload) or `std::string_view`? Recommend packed `uint64_t`.
4. Default `OMP_NUM_THREADS` for short-queue jobs — 32 or fewer? Start at 32 and tune from Phase 4 data.
5. Should the C++ backend also expose `state_digest` for runtime invariance checks, or only at the Python wrapper layer? Recommend C++ exposes it; Python wrapper checks at sampled positions only.
