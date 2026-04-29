#!/bin/bash
# Phase 4: Path A PPM-D C++ backend CPU benchmark — SLURM submission script.
#
# Routing: jobs with --time <= 03:59:00 land in cpu_short automatically. We
# deliberately omit --partition, --gres, and --qos so we never override that
# default (matches the plan and the workspace AGENTS.md guidance).
#
# Submission:
#   sbatch scripts/ppmd_cpp/bench_cpu.sbatch
# Confirm short-queue placement:
#   squeue --me -o '%i %P %T %M %l %R'
# Inspect results after completion:
#   cat results/ppmd_cpp_bench/slurm_<JOBID>_synthetic.json
#
# The body activates .venv-smoke (Python 3.12, where _ppmd_cpp is built) using
# the absolute path to the interpreter so we do not depend on PWD or module
# state.
#
#SBATCH --job-name=ppmd_bench_cpu
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=results/ppmd_cpp_bench/slurm_%j.out
#SBATCH --error=results/ppmd_cpp_bench/slurm_%j.err

set -euo pipefail

REPO_ROOT="/hpfs/scratch/gpfs/mcclec07/parameter_golf2"
VENV_PY="${REPO_ROOT}/.venv-smoke/bin/python"
BENCH_PY="${REPO_ROOT}/scripts/ppmd_cpp/bench_cpu.py"
RESULTS_DIR="${REPO_ROOT}/results/ppmd_cpp_bench"

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

"${VENV_PY}" "${BENCH_PY}" \
    --mode synthetic \
    --positions 4096 \
    --vocab 8192 \
    --threads "${OMP_NUM_THREADS}" \
    --results-dir "${RESULTS_DIR}" \
    --results-name "slurm_${SLURM_JOB_ID}_synthetic"
