#!/bin/bash
# Phase 5: Path A PPM-D C++ backend real-slice eval — SLURM submission script.
#
# Routing: jobs with --time <= 03:59:00 land in cpu_short automatically. We
# deliberately omit --partition, --gres, and --qos so we never override that
# default (matches the plan and AGENTS.md guidance).
#
# Submission:
#   sbatch scripts/ppmd_cpp/eval_real_slice.sbatch
#
# Validate placement without submitting:
#   sbatch --test-only scripts/ppmd_cpp/eval_real_slice.sbatch
#   # Expect output mentioning the cpu_short partition.
#
# Confirm short-queue placement after submission:
#   squeue --me -o '%i %P %T %M %l %R'
#
# Inspect results after completion:
#   cat results/ppmd_cpp_eval/slurm_<JOBID>_eval.json
#   tail -n +1 results/ppmd_cpp_eval/slurm_<JOBID>.{out,err}
#
# What this job does:
#   * Activates .venv-smoke (Python 3.12, where _ppmd_cpp is built) via the
#     absolute path to its interpreter (no PATH/module dependence).
#   * Sets OMP_NUM_THREADS to the SLURM-allocated CPU count.
#   * Fast-fails if _ppmd_cpp is not importable (exit 4).
#   * Invokes scripts/eval_path_a_ppmd.py with --backend cpp on a 5,000-position
#     prefix slice, with a 64-position --backend-equiv-check that asserts
#     |bpb_py - bpb_cpp| <= 1e-10.
#
#SBATCH --job-name=ppmd_eval_real_slice
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=results/ppmd_cpp_eval/slurm_%j.out
#SBATCH --error=results/ppmd_cpp_eval/slurm_%j.err

set -euo pipefail

REPO_ROOT="/hpfs/scratch/gpfs/mcclec07/parameter_golf2"
VENV_PY="${REPO_ROOT}/.venv-smoke/bin/python"
EVAL_PY="${REPO_ROOT}/scripts/eval_path_a_ppmd.py"
RESULTS_DIR="${REPO_ROOT}/results/ppmd_cpp_eval"

mkdir -p "${RESULTS_DIR}"

# OpenMP threads = SLURM-allocated CPU count.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
# Make sure Python can find the built _ppmd_cpp extension.
export PYTHONPATH="${REPO_ROOT}/scripts/ppmd_cpp${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

# Fast-fail precheck: if _ppmd_cpp isn't importable, exit before consuming the
# rest of the SLURM allocation. Returns exit code 4 for a build/import failure.
"${VENV_PY}" -c "import sys; sys.path.insert(0, '${REPO_ROOT}/scripts/ppmd_cpp'); import _ppmd_cpp; print('precheck_ok', _ppmd_cpp.version())" \
    || { echo "FAIL: _ppmd_cpp not importable from ${VENV_PY}; rebuild with 'make -C scripts/ppmd_cpp'"; exit 4; }

"${VENV_PY}" "${EVAL_PY}" \
    --backend cpp \
    --positions 5000 \
    --backend-equiv-check 64 \
    --results-name "slurm_${SLURM_JOB_ID}_eval"
