# `_ppmd_cpp` — Path A PPM-D C++ Backend

Phase 1 stub of the C++ reference backend for the Path A token-normalized PPM-D
evaluator. See [`plans/path-a-ppmd-cpp-backend-plan.md`](../../plans/path-a-ppmd-cpp-backend-plan.md)
for the full multi-phase plan. This phase only establishes the build path and
exposes a sentinel `version()` function — no PPM-D logic yet.

## Prerequisites

- `.venv-smoke/bin/python` (Python 3.12; the system `/bin/python3.8` lacks
  `Python.h` and **cannot** be used).
- `g++` with C++17 + OpenMP support (verified with GCC 8.5.0).
- `pybind11` installed in the venv:
  ```bash
  .venv-smoke/bin/python -m pip install pybind11
  ```

## Build

From the repository root:

```bash
make -C scripts/ppmd_cpp
```

The Makefile uses `../../.venv-smoke/bin/python` by default; override with
`make PYTHON=/path/to/python` if needed. To inspect the resolved flags:

```bash
make -C scripts/ppmd_cpp print-config
```

The build produces `_ppmd_cpp$(EXT_SUFFIX)` in this directory, e.g.
`_ppmd_cpp.cpython-312-x86_64-linux-gnu.so`.

## Smoke import

```bash
.venv-smoke/bin/python -c \
  'import sys; sys.path.insert(0, "scripts/ppmd_cpp"); \
   import _ppmd_cpp; print(_ppmd_cpp.version())'
# -> 0.0.1
```

## Test

```bash
.venv-smoke/bin/python -m unittest tests.test_ppmd_cpp_smoke -v
```

The two smoke tests skip cleanly when the extension is not built or when run
under the system `/bin/python3.8`; they pass after `make`.

## Clean

```bash
make -C scripts/ppmd_cpp clean
```
