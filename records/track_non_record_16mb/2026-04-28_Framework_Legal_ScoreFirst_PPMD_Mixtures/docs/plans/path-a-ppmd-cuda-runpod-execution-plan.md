## Path A PPM-D CUDA RunPod Execution Plan

This document turns `plans/path-a-ppmd-cuda-backend-plan.md` into an **execution-ready RunPod rehearsal ladder** for the CUDA backend. It is still a plan, not an authorization to spend money: no pod should be launched until the operator explicitly approves the budget and supplies `RUNPOD_API_KEY` in the environment.

The plan assumes the **Phase 1 launcher generalization** from `plans/path-a-ppmd-cuda-backend-plan.md` lands first, because the current RunPod tooling is still H100-only and does not yet expose an explicit GPU-SKU selector.

## Scope and assumptions

- GPU work runs on RunPod, not local SLURM.
- Control path uses the repo’s proven **HTTP-bootstrap** workflow in `scripts/runpod_http_rehearsal.py`.
- Retrieval path is the RunPod HTTPS proxy (`https://{pod_id}-30000.proxy.runpod.net/...`).
- Pod shutdown is dual-guarded:
  - pod-side self-termination via `scripts/pod_selfterm.py`
  - launcher-side `finally` cleanup
- Numerical acceptance stays strict:
  - `double` end-to-end
  - no bf16/tf32/fast-math shortcuts
  - byte-prob equivalence to CPU C++ at `<= 1e-15`
  - end-to-end BPB equivalence at `<= 1e-12`
- The CPU C++ backend remains the oracle.
- `2×A100` is optional and only valid if a real two-GPU execution path exists.

## Proposed pod SKUs

These are the concrete RunPod lanes to target once the launcher supports GPU-SKU selection.

| Lane ID | Proposed pod SKU | GPU count | Primary use | Go/No-go rule |
|---|---|---:|---|---|
| `a100-1x` | Secure Cloud `1×A100 80GB` | 1 | Primary cheap lane for env smoke, build smoke, equivalence, benchmark, and likely full eval | Prefer if it passes equivalence and wins on projected full-eval cost |
| `h100-1x` | Secure Cloud `1×H100 80GB HBM3` | 1 | Fallback/control lane; use when A100 is unavailable, unstable, or worse on cost-per-probe | Use if `a100-1x` loses on availability, stability, or projected cost |
| `a100-2x` | Secure Cloud `2×A100 80GB` | 2 | Optional benchmark/full-eval lane only after a true 2-GPU implementation exists | Reject unless measured speedup is large enough to beat both single-GPU lanes on projected full-eval cost |

### Practical SKU notes

- If RunPod only exposes `A100 40GB` on a given day, it is acceptable for **env smoke**, **build smoke**, and probably **kernel equivalence**, but it should not be the first-choice full-eval lane.
- `h100-1x` is the only lane that maps cleanly to the repo’s current hard-coded `GPU_TYPE = "NVIDIA H100 80GB HBM3"` behavior. `a100-1x` and `a100-2x` require the launcher generalization first.
- Final hardware choice must use **actual `costPerHr` returned by RunPod at pod creation time**, not static estimates.

## Target execution surface

The recommended execution surface after launcher work is a dedicated wrapper:

- `scripts/run_ppmd_cuda_runpod.py`

That wrapper should compile down to `scripts/runpod_http_rehearsal.py` under the hood, but it should expose CUDA-specific flags so the operator does not have to hand-build long `--cmd` strings.

### Target wrapper flags

| Flag | Required | Meaning |
|---|---|---|
| `--gpu-sku {a100-1x,a100-2x,h100-1x}` | yes | Explicit pod lane selector |
| `--mode {env-smoke,build-smoke,kernel-equiv,trie-prefix-256,trie-prefix-1k,bench,full-eval}` | yes | Stage selector; wrapper generates the pod command and artifact list |
| `--branch <name>` | yes | Git branch to check out on-pod |
| `--commit <sha>` | yes | Exact commit to check out on-pod |
| `--max-minutes <N>` | yes | User payload wallclock cap; wrapper passes this through to `PGOLF_MAX_MINUTES` |
| `--results-dir <dir>` | yes | Local retrieval directory on this HPC |
| `--pod-name <name>` | recommended | Stable stage-specific pod display name |
| `--runtime-timeout-sec 600` | recommended | Runtime boot timeout for RunPod startup |
| `--docker-image matotezitanka/proteus-pytorch:community` | recommended | Known-good community image |
| `--prefix-positions <N>` | mode-dependent | Prefix length for trie-prefix rehearsals |
| `--download <name>...` | optional | Override artifact list; default should be mode-specific |

### Underlying base-launcher mapping

The wrapper should generate a call to `scripts/runpod_http_rehearsal.py` with these base flags:

- `--gpus {1|2}`
- `--pod-name <stage-name>`
- `--max-minutes <N>`
- `--results-dir <dir>`
- `--cmd '<generated shell payload>'`
- `--download <artifact names...>`
- `--docker-image matotezitanka/proteus-pytorch:community`
- `--runtime-timeout-sec 600`

For the initial execution-ready plan, **do not use `--ssh-upload`** unless a later stage proves a genuinely large artifact must be pushed after pod boot. Prefer on-pod `git clone` / `git checkout` instead.

## Standard artifact contract

Every pod run must retrieve the common artifacts below before teardown.

### Common artifacts for every RunPod stage

- `status.txt`
- `early_status.txt` (optional)
- `pgolf_exit_code.txt`
- `overall_exit_code.txt`
- `pgolf_stdout.txt`
- `http_server.log`
- `launcher_state.json`
- `nvidia_smi.txt`
- `python_version.txt`
- `git_rev.txt`
- `pip_freeze.txt`
- `nvcc_version.txt`

### Stage-specific artifacts

| Stage | Required artifacts |
|---|---|
| env smoke | `cuda_env_probe.json` |
| build smoke | `ppmd_cuda_build.log`, `ppmd_cuda_import_smoke.json`, `ppmd_cuda_build_manifest.txt` |
| kernel equivalence | `ppmd_cuda_kernel_equiv.json`, `ppmd_cuda_kernel_equiv.log` |
| trie prefix 256 | `path_a_cuda_prefix_256.json`, `path_a_cuda_prefix_256.log` |
| trie prefix 1k | `path_a_cuda_prefix_1k.json`, `path_a_cuda_prefix_1k.log` |
| benchmark | `ppmd_cuda_bench_4096x8192.json`, `ppmd_cuda_bench_prefix_1k.json` |
| full eval | `path_a_cuda_full_eval.json`, `path_a_cuda_full_eval.log`, `path_a_cuda_full_eval.sha256` |

### Required local results directory layout

Use one directory per rehearsal stage:

- `results/ppmd_cuda_runpod/01_env_smoke_<lane>/`
- `results/ppmd_cuda_runpod/02_build_smoke_<lane>/`
- `results/ppmd_cuda_runpod/03_kernel_equiv_<lane>/`
- `results/ppmd_cuda_runpod/04_trie_prefix_256_<lane>/`
- `results/ppmd_cuda_runpod/05_trie_prefix_1k_<lane>/`
- `results/ppmd_cuda_runpod/06_bench_<lane>/`
- `results/ppmd_cuda_runpod/07_full_eval_<lane>/`

## Rehearsal ladder

Do not skip stages. Each stage must retrieve artifacts successfully before the next stage is allowed.

### Stage 0 — local preflight (no pod)

**Purpose:** avoid paying for obviously broken code.

**No pod SKU.**

**Expected local checks:**

- current CPU C++ tests stay green
- new RunPod launcher parser/dry-run tests pass
- new CUDA smoke test files at least import/compile cleanly once added

**Local output directory:**

- `results/ppmd_cuda_runpod/00_local_preflight/`

**Acceptance criteria:**

- no syntax/import failures
- no failing CPU reference tests
- no secret leakage in stdout/stderr or generated artifacts

### Stage 1 — environment smoke

**Primary lane:** `a100-1x`

**Fallback lane:** `h100-1x`

**Proposed flags:**

- `--mode env-smoke`
- `--gpu-sku a100-1x`
- `--pod-name ppmd-cuda-env-a100-1x`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--max-minutes 8`
- `--results-dir results/ppmd_cuda_runpod/01_env_smoke_a100_1x`
- `--runtime-timeout-sec 600`
- `--docker-image matotezitanka/proteus-pytorch:community`

**On-pod actions:**

1. `git clone` or fetch the prepared repo state into `/root/rehearsal_src/parameter_golf2`
2. `git checkout <sha>`
3. Write:
   - `nvidia_smi.txt`
   - `python_version.txt`
   - `nvcc_version.txt`
   - `git_rev.txt`
   - `pip_freeze.txt`
   - `cuda_env_probe.json`

**Stage-specific acceptance criteria:**

- `status.txt = DONE`
- `pgolf_exit_code.txt = 0`
- `nvidia_smi.txt` reports the requested lane GPU
- `nvcc_version.txt` exists and is non-empty
- `cuda_env_probe.json` reports:
  - `device_count == 1`
  - `cuda_available == true`
  - `gpu_name` consistent with the selected lane
- `git_rev.txt` matches the requested commit SHA

### Stage 2 — build smoke

**Primary lane:** same winner from Stage 1; default `a100-1x`

**Proposed flags:**

- `--mode build-smoke`
- `--gpu-sku a100-1x`
- `--pod-name ppmd-cuda-build-a100-1x`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--max-minutes 12`
- `--results-dir results/ppmd_cuda_runpod/02_build_smoke_a100_1x`

**On-pod actions:**

1. Repeat source checkout
2. Run the CUDA build target (`make cuda` once it exists)
3. Run a smoke import of the built extension
4. Write:
   - `ppmd_cuda_build.log`
   - `ppmd_cuda_import_smoke.json`
   - `ppmd_cuda_build_manifest.txt`

**Stage-specific acceptance criteria:**

- build exits 0
- `ppmd_cuda_build.log` contains no linker or unresolved-symbol failures
- `ppmd_cuda_import_smoke.json` reports:
  - module import succeeded
  - `cuda_available == true`
  - version string present
- the built module name and hash are recorded in `ppmd_cuda_build_manifest.txt`

### Stage 3 — kernel equivalence

**Primary lane:** same single-GPU lane that passed build smoke

**Proposed flags:**

- `--mode kernel-equiv`
- `--gpu-sku a100-1x`
- `--pod-name ppmd-cuda-kernel-a100-1x`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--max-minutes 15`
- `--results-dir results/ppmd_cuda_runpod/03_kernel_equiv_a100_1x`

**On-pod actions:**

1. Run the targeted kernel equivalence suite against the CPU C++ reference
2. Write:
   - `ppmd_cuda_kernel_equiv.log`
   - `ppmd_cuda_kernel_equiv.json`

**Required JSON fields:**

- `contexts_tested`
- `bytes_tested`
- `max_abs_diff`
- `max_sum_prob_error`
- `gpu_name`
- `git_commit`

**Stage-specific acceptance criteria:**

- `contexts_tested >= 200`
- `bytes_tested >= 51200`
- `max_abs_diff <= 1e-15`
- `max_sum_prob_error <= 1e-15`
- no NaNs/Infs in any output

### Stage 4 — trie-prefix equivalence, 256 positions

**Primary lane:** same single-GPU lane

**Proposed flags:**

- `--mode trie-prefix-256`
- `--gpu-sku a100-1x`
- `--pod-name ppmd-cuda-prefix256-a100-1x`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--prefix-positions 256`
- `--max-minutes 20`
- `--results-dir results/ppmd_cuda_runpod/04_trie_prefix_256_a100_1x`

**On-pod actions:**

1. Run `scripts/eval_path_a_ppmd.py` through the CUDA-enabled backend on a 256-position prefix
2. Enable `--backend-equiv-check 64`
3. Write:
   - `path_a_cuda_prefix_256.log`
   - `path_a_cuda_prefix_256.json`

**Required JSON fields:**

- `positions`
- `backend`
- `backend_equiv_check_positions`
- `backend_equiv_bpb_abs_diff`
- `path_a_score.bpb`
- `path_a_score.positions`
- `missing_positions`

**Stage-specific acceptance criteria:**

- `positions == 256`
- `path_a_score.positions == 256`
- `backend == "cpp"` or explicit CUDA-enabled backend label
- `backend_equiv_check_positions == 64`
- `backend_equiv_bpb_abs_diff <= 1e-12`
- `missing_positions == 0`
- `path_a_score.bpb` finite

### Stage 5 — trie-prefix equivalence, 1k positions

**Primary lane:** same single-GPU lane

**Proposed flags:**

- `--mode trie-prefix-1k`
- `--gpu-sku a100-1x`
- `--pod-name ppmd-cuda-prefix1k-a100-1x`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--prefix-positions 1000`
- `--max-minutes 25`
- `--results-dir results/ppmd_cuda_runpod/05_trie_prefix_1k_a100_1x`

**On-pod actions:**

1. Repeat the prefix run at 1000 positions
2. Write:
   - `path_a_cuda_prefix_1k.log`
   - `path_a_cuda_prefix_1k.json`

**Stage-specific acceptance criteria:**

- `path_a_score.positions == 1000`
- `backend_equiv_bpb_abs_diff <= 1e-12`
- no divergence in sampled state digests if emitted
- no out-of-memory or watchdog timeout
- the run produces enough timing information to extrapolate whether benchmark mode is worth continuing

### Stage 6 — benchmark matrix

**Mandatory lanes:** `a100-1x`, `h100-1x`

**Optional lane:** `a100-2x` only if a real two-GPU path exists

**Proposed flags for each lane:**

- `--mode bench`
- `--gpu-sku <lane>`
- `--pod-name ppmd-cuda-bench-<lane>`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--max-minutes 25`
- `--results-dir results/ppmd_cuda_runpod/06_bench_<lane>`

**Benchmark workload:**

- synthetic `4096 × 8192` throughput case
- same prefix smoke workload used for equivalence sanity

**Required artifacts:**

- `ppmd_cuda_bench_4096x8192.json`
- `ppmd_cuda_bench_prefix_1k.json`

**Required benchmark JSON fields:**

- `gpu_sku`
- `gpu_name`
- `gpu_count`
- `cost_per_hr`
- `positions`
- `vocab`
- `probes_per_second`
- `projected_full_eval_seconds`
- `projected_full_eval_cost`
- `git_commit`

**Stage-specific acceptance criteria:**

- benchmark JSON exists and is non-empty for both mandatory lanes
- `probes_per_second > 0`
- `projected_full_eval_seconds > 0`
- `projected_full_eval_cost > 0`
- the same code revision is used across all lanes

**Decision rule:**

- pick the lane with the **lowest `projected_full_eval_cost`** among lanes that already passed the equivalence rehearsals
- if `a100-2x` is tested, it must show both:
  - measured speedup `>= 1.6×` versus `a100-1x`
  - lower `projected_full_eval_cost` than both `a100-1x` and `h100-1x`
- otherwise reject `a100-2x` and stay single-GPU

### Stage 7 — full non-record eval

**Chosen lane:** winner from Stage 6

**Proposed flags:**

- `--mode full-eval`
- `--gpu-sku <winning-lane>`
- `--pod-name ppmd-cuda-full-<lane>`
- `--branch <prepared-branch>`
- `--commit <sha>`
- `--max-minutes 90`
- `--results-dir results/ppmd_cuda_runpod/07_full_eval_<lane>`

If Stage 6 projects a runtime greater than 90 minutes, raise `--max-minutes` only after explicit approval.

**On-pod actions:**

1. Re-run the short 64-position backend equivalence check before the full pass
2. Run full Path A eval through the CUDA backend
3. Write:
   - `path_a_cuda_full_eval.log`
   - `path_a_cuda_full_eval.json`
   - `path_a_cuda_full_eval.sha256`

**Required JSON fields:**

- `backend`
- `gpu_sku`
- `gpu_name`
- `cost_per_hr`
- `positions`
- `path_a_score.bpb`
- `backend_equiv_check_positions`
- `backend_equiv_bpb_abs_diff`
- `elapsed_seconds`
- `git_commit`

**Stage-specific acceptance criteria:**

- `status.txt = DONE`
- `pgolf_exit_code.txt = 0`
- `positions` equals the full target evaluation length
- `path_a_score.bpb` is finite
- `backend_equiv_check_positions == 64`
- `backend_equiv_bpb_abs_diff <= 1e-12`
- `path_a_cuda_full_eval.sha256` matches the retrieved JSON/log payloads
- all required artifacts are retrieved before termination

## Concrete pod command responsibilities

The wrapper-generated pod command should always do these things in order:

1. create `/root/rehearsal_src` and `/root/rehearsal_out`
2. clone/fetch repo state into `/root/rehearsal_src/parameter_golf2`
3. `git checkout <sha>`
4. capture environment metadata (`nvidia-smi`, `python3 --version`, `nvcc --version`, `pip freeze`, `git rev-parse HEAD`)
5. execute the stage payload
6. write stage-specific JSON/log artifacts under `/root/rehearsal_out/`
7. leave the HTTP server alive long enough for retrieval

## Budget and escalation rules

- No stage may start until the previous stage’s artifacts were retrieved and reviewed locally.
- No `h100-1x` launch is allowed if `a100-1x` has not first failed on availability, stability, or projected cost.
- No `a100-2x` launch is allowed until:
  - single-GPU equivalence is green
  - a real two-GPU code path exists
  - the operator approves the extra spend
- No full-eval launch is allowed until both mandatory benchmark lanes have been compared, unless only one lane is actually available that day.

## Immediate implementation checklist for the launcher work

Before this execution plan can be used, the following implementation items must exist:

1. `scripts/runpod_safe.py` must support explicit GPU-SKU selection instead of H100-only hard-coding.
2. `scripts/runpod_http_rehearsal.py` must record actual `costPerHr` and surface it in the results.
3. A dedicated wrapper (`scripts/run_ppmd_cuda_runpod.py`) should generate the stage commands and default `--download` lists above.
4. The wrapper must write lane ID, commit SHA, and stage name into every stage-specific JSON artifact.

## Minimal go/no-go summary

- **Go from Stage 1 to Stage 2:** toolchain exists and environment smoke is clean.
- **Go from Stage 2 to Stage 3:** build succeeds and CUDA module imports cleanly.
- **Go from Stage 3 to Stage 4:** kernel equivalence is at CPU-level tolerances.
- **Go from Stage 4 to Stage 5:** prefix BPB matches CPU C++ within `1e-12`.
- **Go from Stage 5 to Stage 7:** chosen lane has the lowest projected full-eval cost among equivalent lanes.
- **Stop immediately:** any stage fails artifact retrieval, numerical equivalence, or deterministic shutdown.
