## Plan: Path A PPM-D CUDA Backend (RunPod follow-on)

The CPU C++ backend completed in `plans/path-a-ppmd-cpp-backend-plan.md` hits ~2.28M probes/s on 32 CPU threads, projecting a full non-record eval at ~3.26M s (~38 days). That makes a GPU backend necessary. This revised plan moves **all GPU compile/smoke/benchmark/eval work to RunPod** instead of local SLURM, using the repository’s proven **HTTP-bootstrap + HTTPS-proxy retrieval + pod-side self-termination** workflow.

This is a planning document only. No paid RunPod pod should be launched from this document without explicit budget approval.

**Constraint inheritance from the CPU plan and repo RunPod guardrails:**
- The C++ backend remains the reference oracle. CUDA must match the already-validated CPU C++ semantics before it is trusted for any benchmark or full eval.
- Numerical contract remains `double` end-to-end, with **no** `--use_fast_math`, no bf16/tf32 shortcuts, and BPB equivalence to **>= 12 decimals** vs the CPU C++ reference. If fused-multiply-add drift appears, tighten compile flags further rather than loosening the contract.
- The sequential-position constraint is unchanged. CUDA parallelism is still **within a position** (byte alphabet, trie traversal, vocab sharding), not across positions.
- The proven RunPod control path from this HPC is the **HTTP-bootstrap launcher** in `scripts/runpod_http_rehearsal.py`, plus retrieval through `https://{pod_id}-{port}.proxy.runpod.net/`. Do not rely on SSH, SCP, `rsync`, or Jupyter upload APIs as the primary workflow.
- `scripts/runpod_safe.py` is currently **H100-only** (`GPU_TYPE = "NVIDIA H100 80GB HBM3"`) and `scripts/runpod_http_rehearsal.py` currently assumes H100-only cost accounting. Supporting **1×A100** and **2×A100** is therefore part of this plan, not an already-available feature.
- Because RunPod upload guardrails are strict, do **not** bundle the entire repo. GPU backend sources should be fetched on-pod via a prepared Git branch / commit checkout, or via a tightly curated minimal source bundle after explicit whitelist review. Do not upload `.git/`, `plans/`, credentials, or unrelated results.
- `1×A100`, `2×A100`, and `1×H100` are all candidates. Choose hardware based on **measured probes/s and actual RunPod `costPerHr` returned by the API**, not on hourly price alone. Cheaper hourly hardware is only better if it is cheaper **per projected full eval**.
- `2×A100` is only worth considering after a real two-GPU execution path exists. If the implementation remains single-GPU-per-position, `2×A100` is just twice the bill with extra optimism glitter.

**Phases (6)**

1. **Phase 1: Generalize the RunPod launcher for A100/H100 selection**
   - **Objective:** Extend the existing RunPod launcher stack so Path A CUDA work can target `1×A100`, `2×A100`, or `1×H100` instead of the current H100-only path.
   - **Files/Functions to Modify/Create:** `scripts/runpod_safe.py`, `scripts/runpod_http_rehearsal.py`, and optionally a focused wrapper such as `scripts/run_ppmd_cuda_runpod.py` so the CUDA workflow does not overload the training launcher UX.
   - **Tests to Write:** parser / dry-run tests covering GPU SKU selection, H100 backward compatibility, and correct recording of returned `costPerHr` without exposing secrets.
   - **Steps:**
        1. Replace the hard-coded H100 GPU selection in `scripts/runpod_safe.py` with an explicit GPU-SKU mapping for at least `1×A100`, `2×A100`, and `1×H100`.
        2. Replace the static H100-only cost estimate path in `scripts/runpod_http_rehearsal.py` with **actual RunPod `costPerHr`** captured from pod creation.
        3. Add a source-sync mode suitable for backend development: preferred path is `git clone` / `git checkout` of a prepared branch or commit inside the pod boot command, avoiding large repo uploads and keeping within RunPod transfer guardrails.
        4. Preserve the existing HTTP-bootstrap safety properties: launcher state, early readiness capture, pod-side self-termination, and deterministic teardown in `finally`.

2. **Phase 2: Single-GPU RunPod build and retrieval rehearsal**
   - **Objective:** Prove the cheapest viable single-GPU lane can clone the source, compile the CUDA extension, import it, and retrieve artifacts back to this HPC.
   - **Files/Functions to Modify/Create:** `scripts/ppmd_cpp/Makefile` (CUDA target), RunPod wrapper/command spec for the build, `tests/test_ppmd_cpp_cuda_smoke.py`.
   - **Tests to Write:** `test_ppmd_cuda_extension_imports`, `test_ppmd_cuda_runtime_available`, plus a dry-run launcher test if a new wrapper is added.
   - **Steps:**
        1. Keep local CPU/unit tests as the first gate; do not spend GPU dollars on obviously broken code.
        2. Launch a **single-GPU** smoke on the cheapest candidate lane (`1×A100` if available and cheaper than `1×H100`, otherwise `1×H100`).
        3. On the pod, run: environment probe (`nvidia-smi`, `nvcc --version`, Python import checks), source checkout, `make cuda`, and smoke import of the extension.
        4. Retrieve build logs, smoke-test logs, extension hash/version, and launcher metadata before termination.
   - **Acceptance:** At least one low-cost GPU lane completes source checkout, CUDA build, import smoke, and artifact retrieval with exit code 0.

3. **Phase 3: CUDA byte-prob kernel + single-GPU equivalence on RunPod**
   - **Objective:** Port the PPM-D byte-prob / backoff core to CUDA and validate it on a single RunPod GPU against the CPU C++ reference.
   - **Files/Functions to Modify/Create:** `scripts/ppmd_cpp/src/cuda/backoff_kernel.cuh`, `scripts/ppmd_cpp/src/cuda/ppmd_kernel.cu`, `scripts/ppmd_cpp/src/module.cpp` bindings.
   - **Tests to Write:** `test_ppmd_cuda_byte_prob_matches_cpp_reference` (200 contexts × 256 bytes, diff ≤ `1e-15` in double), `test_ppmd_cuda_byte_probs_sums_to_one`.
   - **Steps:**
        1. Decide device context-store layout: perfect hash, sorted flat arrays + lookup, or another deterministic layout. Perfect hash remains the preferred starting point for bounded order=5 / SP8192.
        2. Implement the byte-prob kernel without fast math shortcuts.
        3. Run the equivalence suite on the cheapest passing single-GPU lane from Phase 2.
        4. Only if `1×A100` shows numerical or tooling issues should the default shift to `1×H100` for this phase.
   - **Acceptance:** Byte-prob outputs are numerically indistinguishable from CPU C++ within the existing test tolerances, and logs/results are retrieved successfully.

4. **Phase 4: CUDA trie scorer + single-GPU end-to-end Path A**
   - **Objective:** Port `trie_partial_z_and_target` and `score_path_a_arrays` to CUDA, still using a single GPU first, and verify end-to-end BPB against the CPU C++ reference.
   - **Files/Functions to Modify/Create:** `scripts/ppmd_cpp/src/cuda/trie_kernel.cu`, `scripts/ppmd_cpp/src/cuda/scorer_kernel.cu`, `scripts/eval_path_a_ppmd.py` backend dispatch if needed.
   - **Tests to Write:** `test_ppmd_cuda_trie_partial_matches_cpp_single_shard`, `test_ppmd_cuda_score_path_a_arrays_matches_cpp` (BPB diff ≤ `1e-12` end-to-end on synthetic prefixes), plus a short RunPod prefix smoke that retrieves JSON output.
   - **Steps:**
        1. Keep the outer position loop on host; move the per-position trie work to device kernels.
        2. Preserve deterministic reduction ordering. If GPU reductions introduce drift, fix the reduction order rather than weakening the acceptance bar.
        3. Run a short prefix eval through `scripts/eval_path_a_ppmd.py --backend cpp --backend-equiv-check 64` using the CUDA-enabled extension path.
   - **Acceptance:** Single-GPU CUDA path matches CPU C++ on the prefix smoke and passes the targeted equivalence tests.

5. **Phase 5: RunPod benchmark matrix and cost-effectiveness gate**
   - **Objective:** Measure the real throughput/cost tradeoff of `1×A100`, `1×H100`, and, if implemented, `2×A100`, then choose the cheapest viable hardware for full eval.
   - **Files/Functions to Modify/Create:** `scripts/ppmd_cpp/bench_gpu.py`, RunPod launcher wrapper or command presets, and result JSON schemas capturing GPU SKU and actual `costPerHr`.
   - **Tests to Write:** `test_bench_gpu_cli_parses`, `test_bench_gpu_writes_results_json`.
   - **Steps:**
        1. Benchmark `1×A100` and `1×H100` on the same synthetic workload and prefix smoke.
        2. Compute for each configuration:
           - measured `probes_per_second`
           - `projected_full_eval_seconds`
           - `projected_full_eval_cost = costPerHr × projected_full_eval_seconds / 3600`
        3. Only benchmark `2×A100` **if** a real two-GPU execution path exists (for example, deterministic per-position vocab-shard split across devices with host or NCCL reduction).
        4. Choose the winner by **lowest projected full-eval cost among configurations that pass equivalence and fit the agreed wallclock budget**.
   - **Decision rule:**
        - Prefer `1×A100` if it passes equivalence and yields lower projected full-eval cost than `1×H100`.
        - Prefer `2×A100` only if measured speedup is large enough to beat both `1×A100` and `1×H100` on projected full-eval cost; do not assume this in advance.
        - Fall back to `1×H100` if A100 lanes are unavailable, unstable, or worse on cost-per-probe.

6. **Phase 6: Full non-record Path A eval on the chosen RunPod SKU**
   - **Objective:** Run the full real-data Path A eval through the CUDA backend on the hardware selected in Phase 5.
   - **Files/Functions to Modify/Create:** a dedicated RunPod eval wrapper/command preset, plus any result JSON/log naming needed for retrieval.
   - **Tests to Write:** integration only; the final gate is successful prefix equivalence + full artifact retrieval.
   - **Steps:**
        1. Re-run the short prefix equiv check on the exact chosen pod shape before the full run.
        2. Launch the bounded full eval with explicit retrieval buffer and pod-side self-termination.
        3. Retrieve JSON outputs, logs, launcher state, and any benchmark metadata before teardown.
        4. If no single-GPU RunPod lane is cost-effective enough, open a separate multi-GPU CUDA execution plan rather than quietly escalating scope here.

**Open Questions**
1. What are the exact RunPod GPU identifiers and real `costPerHr` values for the best available `1×A100`, `2×A100`, and `1×H100` lanes at launch time?
2. Should the CUDA backend continue to live under `_ppmd_cpp` with runtime GPU detection, or split into `_ppmd_cuda` to keep the import/ABI surface simpler?
3. Is a Git-based source checkout sufficient for backend-development pods, or do we need a separate reviewed minimal source-bundle path for unmerged local changes?
4. What deterministic cross-device reduction strategy should be used if `2×A100` is pursued: host reduction, NCCL all-reduce, or explicit shard-gather on CPU?
5. Can multiple positions ever be safely batched on GPU without violating the sequential PPM update semantics, or should all optimization effort stay within a single position forever?
